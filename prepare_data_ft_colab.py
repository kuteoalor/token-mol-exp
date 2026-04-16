#!/usr/bin/env python3
"""Build Token-Mol fine-tuning data.

This script is Colab-friendly and creates the 4 pkl files expected by
`pocket_fine_tuning_rmse.py`:
  - mol_input.pkl
  - train_protein_represent.pkl
  - val_mol_input.pkl
  - val_protein_represent.pkl

Typical usage:
python prepare_data_ft_colab.py \
  --sdf_dir ./content/vs_hits \
  --pocket_embedding_pkl ./encoder/embeddings/LGH_B1_pocket.pkl \
  --author_mol_input_pkl ./data/mol_input.pkl \
  --author_train_protein_pkl ./data/train_protein_represent.pkl \
  --author_val_mol_input_pkl ./data/val_mol_input.pkl \
  --author_val_protein_pkl ./data/val_protein_represent.pkl \
  --output_dir ./content/data_ft
"""

import argparse
import gzip
import pickle
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from tqdm.auto import tqdm

from utils.standardization import get_torsion_angles


def read_data(path: Path) -> List:
    """Read pkl file(s) written as one-or-many sequential pickle dumps."""
    data = []
    with path.open("rb") as f:
        while True:
            try:
                chunk = pickle.load(f)
                if isinstance(chunk, list):
                    data.extend(chunk)
                else:
                    data.append(chunk)
            except EOFError:
                break
    return data


def write_single_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_vocab(vocab_path: Path) -> set:
    vocab = set()
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                vocab.add(token)
    return vocab


def iter_sdf_mols(path: Path) -> Iterable[Optional[Chem.Mol]]:
    """Yield molecules from .sdf or .sdf.gz."""
    suffixes = path.suffixes
    if suffixes[-2:] == [".sdf", ".gz"] or path.name.endswith(".sdf.gz"):
        with gzip.open(path, "rb") as fh:
            supplier = Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=False)
            for mol in supplier:
                yield mol
    elif path.suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        for mol in supplier:
            yield mol
    else:
        return


def tokenize_smiles(smiles: str) -> List[str]:
    """Tokenize smiles to match Token-Mol vocab style.

    - bracket expressions [..] are kept as one token
    - Br and Cl are two-character tokens
    - remaining content is split character-wise
    """
    bracket_regex = r"(\[[^\[\]]{1,16}\])"
    chunks = re.split(bracket_regex, smiles)
    tokens: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if chunk.startswith("[") and chunk.endswith("]"):
            tokens.append(chunk)
            continue
        i = 0
        while i < len(chunk):
            pair = chunk[i : i + 2]
            if pair in ("Br", "Cl"):
                tokens.append(pair)
                i += 2
            else:
                tokens.append(chunk[i])
                i += 1
    return tokens


def canonicalize_and_reorder_3d(mol_3d: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str], str]:
    """Return 3D mol reordered to canonical smiles atom order."""
    if mol_3d is None:
        return None, None, "mol_none"
    if mol_3d.GetNumConformers() == 0:
        return None, None, "no_conformer"

    try:
        mol_no_h = Chem.RemoveHs(mol_3d, sanitize=False)
        Chem.SanitizeMol(mol_no_h)
    except Exception:
        return None, None, "remove_hs_or_sanitize_failed"

    try:
        smiles = Chem.MolToSmiles(mol_no_h, canonical=True)
        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_smiles is None:
            return None, None, "mol_from_smiles_failed"
    except Exception:
        return None, None, "smiles_build_failed"

    match = mol_no_h.GetSubstructMatch(mol_smiles)
    if not match or len(match) != mol_smiles.GetNumAtoms():
        try:
            # Fallback: force bond order assignment from the canonical template.
            mol_with_bond_orders = AllChem.AssignBondOrdersFromTemplate(mol_smiles, mol_no_h)
            match = mol_with_bond_orders.GetSubstructMatch(mol_smiles)
            if not match or len(match) != mol_smiles.GetNumAtoms():
                return None, None, "atom_mapping_failed"
            mol_ordered = Chem.RenumberAtoms(mol_with_bond_orders, list(match))
        except Exception:
            return None, None, "atom_mapping_failed"
    else:
        mol_ordered = Chem.RenumberAtoms(mol_no_h, list(match))

    try:
        canonical_check = Chem.MolToSmiles(mol_ordered, canonical=True)
        if canonical_check != smiles:
            return None, None, "canonical_reorder_mismatch"
    except Exception:
        return None, None, "canonical_reorder_check_failed"

    if mol_ordered.GetNumConformers() == 0:
        return None, None, "no_conformer_after_reorder"
    return mol_ordered, smiles, "ok"


def mol_to_training_string(
    mol_3d: Chem.Mol,
    vocab: set,
    max_total_tokens: int,
    torsion_decimals: int,
) -> Tuple[Optional[str], str]:
    """Convert one 3D RDKit mol to '<token...> GEO <torsions...>'."""
    mol_ordered, smiles, status = canonicalize_and_reorder_3d(mol_3d)
    if mol_ordered is None or smiles is None:
        return None, status

    torsions = get_torsion_angles(mol_ordered)
    if not torsions:
        return None, "no_rotatable_bonds"

    conf = mol_ordered.GetConformer(0)
    values: List[str] = []
    for atom_idx in torsions:
        angle_rad = rdMolTransforms.GetDihedralRad(
            conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]
        )
        values.append(f"{angle_rad:.{torsion_decimals}f}")

    smiles_tokens = tokenize_smiles(smiles)
    if any(tok not in vocab for tok in smiles_tokens):
        return None, "smiles_token_oov"
    if any(val not in vocab for val in values):
        return None, "torsion_token_oov"

    seq_tokens = smiles_tokens + ["GEO"] + values
    # Account for wrappers in data_loader:
    # <|beginoftext|> <|mask:0|> <|mask:0|> + seq + <|endofmask|>
    total_tokens_after_wrapper = len(seq_tokens) + 4
    if total_tokens_after_wrapper > max_total_tokens:
        return None, "too_long"

    return " ".join(seq_tokens), "ok"


def list_sdf_files(sdf_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.sdf.gz" if recursive else "*.sdf.gz"
    files = sorted(sdf_dir.glob(pattern))
    # Also support non-gz .sdf when present.
    pattern_sdf = "**/*.sdf" if recursive else "*.sdf"
    files.extend(sorted(sdf_dir.glob(pattern_sdf)))
    return files


def load_single_pocket_representation(path: Path) -> np.ndarray:
    entries = read_data(path)
    if not entries:
        raise ValueError(f"No entries found in pocket embedding file: {path}")
    pocket = np.asarray(entries[0], dtype=np.float32)
    return pocket


def build_custom_data(
    sdf_files: Sequence[Path],
    vocab: set,
    max_total_tokens: int,
    torsion_decimals: int,
) -> Tuple[List[str], Dict[str, int]]:
    custom_mols: List[str] = []
    counts: Dict[str, int] = {}
    for sdf_file in tqdm(sdf_files, desc="Reading SDF files"):
        for mol in iter_sdf_mols(sdf_file):
            line, status = mol_to_training_string(
                mol_3d=mol,
                vocab=vocab,
                max_total_tokens=max_total_tokens,
                torsion_decimals=torsion_decimals,
            )
            counts[status] = counts.get(status, 0) + 1
            if line is not None:
                custom_mols.append(line)
    return custom_mols, counts


def split_custom_train_val(
    custom_mols: List[str],
    custom_proteins: List[np.ndarray],
    holdout_fraction: float,
    seed: int,
) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    if holdout_fraction <= 0 or not custom_mols:
        return custom_mols, custom_proteins, [], []

    n = len(custom_mols)
    n_val = int(round(n * holdout_fraction))
    n_val = min(max(n_val, 1), n - 1) if n > 1 else 0
    if n_val == 0:
        return custom_mols, custom_proteins, [], []

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_idx = set(idx[:n_val])

    train_m, train_p, val_m, val_p = [], [], [], []
    for i in range(n):
        if i in val_idx:
            val_m.append(custom_mols[i])
            val_p.append(custom_proteins[i])
        else:
            train_m.append(custom_mols[i])
            train_p.append(custom_proteins[i])
    return train_m, train_p, val_m, val_p


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir", type=Path, required=True, help="Folder with *.sdf.gz (or *.sdf).")
    parser.add_argument(
        "--pocket_embedding_pkl",
        type=Path,
        required=True,
        help="Path to encoder/embeddings/<name>.pkl containing one pocket embedding.",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("./data/torsion_version/torsion_voc_pocket.csv"),
        help="Token-Mol vocabulary path.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./data_ft"), help="Output folder.")

    parser.add_argument("--author_mol_input_pkl", type=Path, default=None)
    parser.add_argument("--author_train_protein_pkl", type=Path, default=None)
    parser.add_argument("--author_val_mol_input_pkl", type=Path, default=None)
    parser.add_argument("--author_val_protein_pkl", type=Path, default=None)

    parser.add_argument(
        "--holdout_fraction",
        type=float,
        default=0.1,
        help="Fraction of custom molecules moved to validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=200,
        help="Maximum encoded length used by pocket_fine_tuning_rmse.py data_loader.",
    )
    parser.add_argument(
        "--torsion_decimals",
        type=int,
        default=2,
        help="Round torsions to this decimal count to match vocab tokens.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search sdf_dir recursively for *.sdf.gz/*.sdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = setup_args()
    if not args.sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir does not exist: {args.sdf_dir}")
    if not args.pocket_embedding_pkl.exists():
        raise FileNotFoundError(f"pocket_embedding_pkl does not exist: {args.pocket_embedding_pkl}")
    if not args.vocab_path.exists():
        raise FileNotFoundError(f"vocab_path does not exist: {args.vocab_path}")
    if not 0 <= args.holdout_fraction < 1:
        raise ValueError("--holdout_fraction must be in [0, 1).")

    sdf_files = list_sdf_files(args.sdf_dir, recursive=args.recursive)
    if not sdf_files:
        raise ValueError(f"No .sdf.gz or .sdf files found in: {args.sdf_dir}")

    vocab = load_vocab(args.vocab_path)
    pocket = load_single_pocket_representation(args.pocket_embedding_pkl)

    custom_mols, stats = build_custom_data(
        sdf_files=sdf_files,
        vocab=vocab,
        max_total_tokens=args.max_total_tokens,
        torsion_decimals=args.torsion_decimals,
    )
    if not custom_mols:
        raise ValueError(
            "No valid custom molecules were built. Check status counts in report below "
            "and verify your SDF poses/token coverage."
        )

    custom_proteins = [pocket.copy() for _ in range(len(custom_mols))]
    custom_train_m, custom_train_p, custom_val_m, custom_val_p = split_custom_train_val(
        custom_mols=custom_mols,
        custom_proteins=custom_proteins,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    author_train_m = read_data(args.author_mol_input_pkl) if args.author_mol_input_pkl else []
    author_train_p = read_data(args.author_train_protein_pkl) if args.author_train_protein_pkl else []
    author_val_m = read_data(args.author_val_mol_input_pkl) if args.author_val_mol_input_pkl else []
    author_val_p = read_data(args.author_val_protein_pkl) if args.author_val_protein_pkl else []

    if author_train_m and author_train_p and len(author_train_m) != len(author_train_p):
        raise ValueError("author train mol/protein lengths mismatch")
    if author_val_m and author_val_p and len(author_val_m) != len(author_val_p):
        raise ValueError("author val mol/protein lengths mismatch")

    train_mol = author_train_m + custom_train_m
    train_protein = author_train_p + custom_train_p
    val_mol = author_val_m + custom_val_m
    val_protein = author_val_p + custom_val_p

    if len(train_mol) != len(train_protein):
        raise ValueError("Final train mol/protein lengths mismatch.")
    if len(val_mol) != len(val_protein):
        raise ValueError("Final val mol/protein lengths mismatch.")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_single_pickle(out / "mol_input.pkl", train_mol)
    write_single_pickle(out / "train_protein_represent.pkl", train_protein)
    write_single_pickle(out / "val_mol_input.pkl", val_mol)
    write_single_pickle(out / "val_protein_represent.pkl", val_protein)

    print("Done.")
    print(f"SDF files scanned: {len(sdf_files)}")
    print(f"Custom valid molecules: {len(custom_mols)}")
    print(f"Custom train/val split: {len(custom_train_m)} / {len(custom_val_m)}")
    print(f"Author train/val: {len(author_train_m)} / {len(author_val_m)}")
    print(f"Final train/val: {len(train_mol)} / {len(val_mol)}")
    print("Status counts:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
