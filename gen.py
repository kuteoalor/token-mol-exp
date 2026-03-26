# -*- coding:utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
import time
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from bert_tokenizer import ExpressionBertTokenizer
from ada_model import Token3D
from pocket_fine_tuning_rmse import Ada_config
from pocket_fine_tuning_rmse import read_data

#from early_stop.pytorchtools import EarlyStopping


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', default='./Pretrained_model', type=str, help='path to pretrained GPT2 folder')
    parser.add_argument('--model_path', default='./Trained_model/pocket_generation.pt', type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_version/torsion_voc_pocket.csv", type=str, help='')
    parser.add_argument('--protein_path', default='./example/ARA2A.pkl', type=str, help='')
    parser.add_argument('--output_path', default='output.csv', type=str, help='')
    parser.add_argument('--batch_size', default=25, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=4, type=int, required=False, help='epochs')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '<|endofmask|>': break
        chars.append(i)
    seq = " ".join(chars)
    return seq

@torch.no_grad()
def predict(model, tokenizer, batch_size, single_pocket,
            text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)
    # text = ""
    protein_batch = single_pocket
    model.to(device)
    model.eval()
    #time1 = time.time()
    max_length = 195
    input_ids = []
    input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    output_vocab_size = model.lm_head.out_features
    input_vocab_size = model.mol_model.transformer.wte.num_embeddings
    safe_vocab_size = min(output_vocab_size, input_vocab_size)
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    unk_id = min(max(int(unk_id), 0), input_vocab_size - 1)

    if len(input_ids) == 0:
        raise ValueError("Prompt produced an empty token list.")
    if max(input_ids) >= input_vocab_size or min(input_ids) < 0:
        input_ids = [tok if 0 <= tok < input_vocab_size else unk_id for tok in input_ids]

    input_length = len(input_ids)

    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)

    Seq_list = []

    eos_token = '<|endofmask|>'
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_ids = tokenizer.encode(eos_token, add_special_tokens=False)
        if len(eos_ids) == 0:
            raise ValueError(f"Could not resolve EOS token id for {eos_token}")
        # Some tokenizers may return repeated ids for a single special token string.
        eos_id = eos_ids[0]
    if eos_id < 0 or eos_id >= input_vocab_size:
        eos_id = unk_id
    finished = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

    protein_batch = torch.tensor(protein_batch, dtype=torch.float32)
    protein_batch = protein_batch.to(device)
    protein_batch = protein_batch.repeat(batch_size, 1, 1)
    for i in range(max_length):
        # Guard against out-of-range token ids before embedding lookup.
        if input_tensor.min().item() < 0 or input_tensor.max().item() >= input_vocab_size:
            input_tensor = input_tensor.clamp(min=0, max=input_vocab_size - 1)
        inputs = input_tensor.to(device)
        outputs = model(inputs, protein_batch)
        step_logits = outputs.logits[:, -1, :safe_vocab_size]
        step_logits = torch.nan_to_num(step_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = F.softmax(step_logits.float(), dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs_sum = probs.sum(dim=1, keepdim=True)

        if (probs_sum <= 0).any():
            last_token_id = torch.argmax(step_logits, dim=1, keepdim=True)
        else:
            probs = probs / probs_sum
            # Sampling on CPU avoids CUDA-side assert failures from multinomial kernels.
            last_token_id = torch.multinomial(probs.detach().cpu(), 1).to(device)
        last_token_id = last_token_id.clamp(min=0, max=input_vocab_size - 1)
        #last_token_id = torch.argmax(logits,1).view(-1,1)

        EOS_sampled = (last_token_id == eos_id)
        finished = finished | EOS_sampled
        if finished.all():
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id.squeeze(-1).detach().cpu().tolist())
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    Seq_list = np.array(Seq_list).T

    return Seq_list

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    model_path, protein_path = args.model_path, args.protein_path
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    model = Token3D(pretrain_path=args.pretrain_path, config=Ada_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Ignoring {len(unexpected_keys)} unexpected checkpoint keys (e.g. {unexpected_keys[:2]})")
    if missing_keys:
        print(f"Missing {len(missing_keys)} model keys when loading checkpoint (e.g. {missing_keys[:2]})")
    eval_data_protein = read_data(args.protein_path)
    print('Model loaded successfully. Starting generation...')
    all_output = []
    # Total number = range * batch size
    for pocket in eval_data_protein:
        one_output = []
        Seq_all = []
        for i in tqdm(range(args.epochs)):
            Seq_list = predict(model, tokenizer, single_pocket=pocket, batch_size=args.batch_size)
            Seq_all.extend(Seq_list)
        for j in Seq_all:
            one_output.append(decode(j))
        all_output.append(one_output)

    print(f'Genetion finished! Raw seqs saves at {args.output_path}')
    output = pd.DataFrame(all_output)

    output.to_csv(args.output_path, index=False, header=False, mode='a')
