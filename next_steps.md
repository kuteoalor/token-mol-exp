
Here is a **step-by-step guide** you can follow end-to-end. It stays on **fine-tuning only** (no RL) and explains how your **virtual-screening `*.sdf.gz`** fits in, in plain language.

---

## 0. What you are actually doing (big picture)

Token-Mol’s pocket model learns: **“given this protein pocket (as a fixed-size vector), predict the next token”** for strings that look like:

**space-separated SMILES tokens + `GEO` + torsion numbers + end marker**

So you are **not** training on “raw SDF files” directly. You must turn each hit into **that text line**, and pair it with **the same pocket encoding** the authors use (ResGen-style embedding from your PDB + reference ligand).

**Important expectation:** A few thousand VS hits **alone** may **overfit** one pocket or give unstable training. The usual approach is **author’s large training set + your ligands** mixed in, not **only** your 2000 molecules.

---

## 1. What you need on disk before you start

| Item | Role |
|------|------|
| **Token-Mol repo** | Code you already have |
| **GPU machine (Linux recommended)** | Training is practical on CUDA; CPU is usually too slow |
| **Pretrained GPT weights** | Folder `Pretrained_model/` from [Zenodo](https://zenodo.org/records/13428828) (README) |
| **Author’s processed training data (strongly recommended)** | `mol_input.pkl`, `train_protein_represent.pkl`, `val_mol_input.pkl`, `val_protein_represent.pkl` from [Zenodo](https://zenodo.org/records/13578841) → put under `data/` as in the README |
| **ResGen checkpoint** | For `encoder/encode.py` (Google Drive link in README) → `encoder/ckpt/` |
| **Your target** | **Receptor PDB** (same structure you screened against) + **reference ligand** in the pocket (SDF/MOL2) for box center — same idea as README for encoding |
| **Your VS hits** | `*.sdf.gz` (poses); you’ll derive SMILES + torsions (below) |

---

## 2. Install the environment

Follow **README** versions (Python 3.8, PyTorch, RDKit, `transformers`, etc.) and, for the encoder, the **PyG / torch-scatter** stack listed there.

You should be able to run:

- `python pocket_fine_tining_rmse.py` (after data is in place) — *typo intentional? file is `pocket_fine_tuning_rmse.py`*
- `python encoder/encode.py ...`

---

## 3. Encode **your** pocket (protein tensor for training and generation)

The training script expects `train_protein_represent.pkl` as a **list** of arrays: **one pocket embedding per training molecule**, same index as each string in `mol_input.pkl`.

**For a single target pocket:**

1. Put your **protein PDB** and **reference ligand SDF** (in the pocket) where `encoder/encode.py` expects them.
2. Run (paths from README pattern):

   ```bash
   python encoder/encode.py --pdb_file YOUR.pdb --sdf_file YOUR_REF_LIGAND.sdf --output_name MYPOCKET
   ```

3. This produces something like `encoder/embeddings/MYPOCKET.pkl` containing **one** protein representation.

**For fine-tuning:** if you add **N** custom ligand strings, you need **N copies of the same pocket array** in the protein list (same pocket, different molecules). Author data already has one protein row per molecule; your custom rows must follow the **same pairing rule**.

---

## 4. Turn `*.sdf.gz` virtual hits into training strings

Each training line (before the script adds the prefix/suffix) must be:

```text
<token> <token> ... <token> GEO <float> <float> ...
```

Rules that match this repo:

- **Tokens are split on spaces** (`bert_tokenizer.py` splits on `' '`).
- **`GEO`** is the literal separator (see `confs_to_mols_pocket.py` / `parse_geo_record`).
- **Torsion values** must be consistent with what **post-processing** expects: `conformer_match` uses `get_torsion_angles` from `utils/standardization.py` and **`apply_changes` / `SetDihedralRad`** — so angles are handled in **radians**, in the **order** returned by `get_torsion_angles` on the **SMILES-derived** molecule.

### 4.1 Practical recipe per molecule

For **each** record in your SDF (decompress with `gzip` or read with RDKit’s supplier on `.sdf.gz` if supported):

1. **Load the 3D molecule** from SDF (`sanitize=True` if it succeeds; if not, try `sanitize=False` and fix or skip).
2. **Canonical SMILES** (or a fixed SMILES you always use):  
   `smi = Chem.MolToSmiles(mol, canonical=True)`  
   then `mol_smiles = Chem.MolFromSmiles(smi)`.
3. **Put the 3D coordinates onto the SMILES atom ordering** (this step matters; VS poses are on the SDF order, SMILES order can differ). Use RDKit alignment / substructure match so the **conformer on `mol_smiles`** matches your pose (if this fails, **skip** the molecule or fix it manually).
4. **Rotatable bonds** (same as the project):  
   `bonds = get_torsion_angles(mol_smiles)`  
   For each bond, compute the dihedral in **radians** from that conformer (the same convention the paper uses for targets — the training CSV uses floats like `1.72`; your builder should match **`conformer_match`** behavior; when in doubt, **round-trip test**: build string → `conformer_match` → reasonable structure).
5. **SMILES → space-separated tokens**  
   Split SMILES into tokens that appear in `data/torsion_version/torsion_voc_pocket.csv`. A common pattern is: bracket groups `[...]` as one token, rest character-by-character (see the idea in `reinforce/MCMG_utils/data_structs.py` `tokenize`). Then join with spaces:  
   `" ".join(tokens)`.
6. **Append torsions** as strings (e.g. `f"{x:.2f}"`), space-separated after `GEO`.

**Skip** molecules with:

- no rotatable bonds (`get_torsion_angles` empty),
- tokens **not in vocab** (replace or drop),
- sequence **too long** after encoding (training uses `max_length=200` in `data_loader` — long ligands may truncate or break; prefer filtering).

Collect all strings into a Python list `my_mol_strings`.

### 4.2 Build `my_protein_list`

Let `P` be the **one** numpy array you loaded from `encoder/embeddings/MYPOCKET.pkl` (the code stores a list with one array per file in the current `encode.py` pattern — use the actual shape your file contains).

Then:

```python
my_protein_list = [P.copy() for _ in range(len(my_mol_strings))]
```

Index `i` of `my_mol_strings` **must** match index `i` of `my_protein_list`.

---

## 5. Merge with the **author’s** data (recommended)

**Why:** A few thousand poses on one pocket are not as diverse as the original dataset; the model may memorize or collapse.

**How:**

1. Load author pickles with the same logic as `read_data` in `pocket_fine_tuning_rmse.py` (loop `pickle.load` until EOF).
2. **Concatenate** lists:

   `mol_data = author_mol + my_mol_strings`  
   `protein_matrix = author_prot + my_protein_list`

3. **Validation set:** either use the author’s `val_*` only, or hold out ~5–10% of **your** strings (and matching protein rows) into `eval_mol` / `eval_protein`.

4. **Save** new pickles (simplest: one list per file):

   ```python
   import pickle
   with open("data/mol_input.pkl", "wb") as f:
       pickle.dump(mol_data, f)
   with open("data/train_protein_represent.pkl", "wb") as f:
       pickle.dump(protein_matrix, f)
   # same for val files if you replace those too
   ```

If you **only** have your pocket, **every** `train_protein_represent` row should still be **your** encoded pocket (repeated).

---

## 6. Point the trainer at your data

`pocket_fine_tuning_rmse.py` **hardcodes**:

- `./data/train_protein_represent.pkl`
- `./data/mol_input.pkl`
- `./data/val_protein_represent.pkl`
- `./data/val_mol_input.pkl`

So either **overwrite** those four files in `data/` with your merged pickles, or **edit the paths** in the `if __name__ == '__main__':` block to your filenames.

Also check:

- `args.model_path = './Pretrained_model'` (must contain the downloaded pretrained GPT).
- **`Ada_config`** `n_positions` / `n_ctx` (e.g. 380) must match **`Pretrained_model/config.json`** so shapes line up.

---

## 7. Run fine-tuning

From the repo root (README-style):

```bash
python pocket_fine_tuning_rmse.py --batch_size 32 --epochs 40 --lr 5e-3 --every_step_save_path Trained_model/pocket_generation
```

**Beginner notes:**

- **First run:** use author defaults; confirm loss prints and a `.pt` appears under `Trained_model/`.
- **Small custom data:** consider **fewer `--epochs`**, **lower `--lr`**, and **lower `--warmup_steps`** (default `20000` is huge if you only have a few thousand steps per epoch — you may want to reduce warmup so learning rate is not stuck near zero for the whole run; this often needs a bit of trial and error).
- **Early stopping** writes to `early_stop_path`; watch validation loss in the console.

Output checkpoint: typically `Trained_model/pocket_generation.pt` (plus epoch saves depending on your edits).

---

## 8. Generate with **your** pocket and **new** weights

1. **Protein input for `gen.py`:** must be the **same format** as training — the list saved from encoding your pocket (often you `pickle.dump([P], f)` or reuse the embedding file in the shape `read_data` expects for `protein_path`).
2. Run `gen.py` with:

   - `--model_path` → your new `pocket_generation.pt`
   - `--protein_path` → pickle pointing at **your** pocket embedding(s)
   - `--vocab_path` → `data/torsion_version/torsion_voc_pocket.csv`

3. Post-process with `confs_to_mols_pocket.py` as in the README.

---

## 9. Sanity checks (worth doing once)

1. **Round-trip:** For a few built strings, run through `conformer_match` (same as `confs_to_mols_pocket.py`) — if it often fails, your torsion or SMILES tokenization is wrong.
2. **Token coverage:** Count `unk` or failed `tokenizer.encode` — drop or fix bad examples.
3. **Length:** If most sequences exceed 200 tokens after wrapping with masks, increase risk of truncation in `data_loader` (would require code/config change to match `n_ctx`).

---

## 10. What “better for my pocket” realistically means

- Fine-tuning **biases** the model toward sequences **like your training lines** (chemistry + torsions + that pocket vector).
- It **does not** guarantee better docking or affinity unless your **labels** (which ligands are “good”) are reflected in **what you put in** — here you only add **poses/SMILES**, not scores. Using **only** random VS hits without the author mix can hurt generalization.

---

If you want this as a **file in the repo** (e.g. `docs/FINETUNING_FROM_VS.md`) or a **small ready-made script** that reads `*.sdf.gz` and writes `mol_input` strings (with the alignment step stubbed), say which you prefer and we can add it in a follow-up.