# -*- coding:utf-8 -*-
# @Author: jikewang
# @Time: 2023/10/12 9:43
# @File: ada_model.py
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2 import GPT2LMHeadModel
import torch.nn as nn
from typing import Optional, Tuple, Union


class CrossSelfAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # Keep compatibility with Token-Mol checkpoints while avoiding private
        # GPT2Attention helpers that changed across transformers versions.
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]
        query = query.view(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, query_len, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs[0]


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, layer_norm: nn.Module, dropout: float = 0.1, ):
        super().__init__()
        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = layer_norm

    def forward(self, x, *args, **kwargs):
        # x is the mol embedding, already normalize
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)

        x = outputs
        x = self.dropout_module(x)
        x = residual + x
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            ffn_embedding_dim: int,
            activation_dropout: float = 0.1,
            max_tokens_per_msa: int = 2 ** 14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class Token3D(nn.Module):
    def __init__(self, pretrain_path, config):
        super(Token3D, self).__init__()
        self.mol_model = GPT2LMHeadModel.from_pretrained(pretrain_path)
        self.CrossSelfAttention = CrossSelfAttention(config=config)
        self.up_sample = nn.Linear(256, config.n_embd)
        self.dropout = nn.Dropout(0.1)
        self.protein_adapter_ffn = ResidualBlock(
            layer=FeedForwardNetwork(
                config.n_embd,
                config.n_embd // 2,  # NOTE: bottleneck FFN is important
                activation_dropout=0.1
            ),
            layer_norm=nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            dropout=0.1,
        )
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, protein_matrix):
        len_protein_info = protein_matrix.size(1)
        x = self.mol_model(x, output_hidden_states=True).hidden_states[-1]

        # res net
        residual = x
        protein_matrix = self.up_sample(protein_matrix)
        protein_matrix = self.ln_1(protein_matrix)

        x = torch.cat((protein_matrix, x), dim=1)  # concat protein info
        x = self.ln_2(x)
        x = self.CrossSelfAttention(x)  # cross-self attention
        x = self.dropout(x)
        x = x[:, len_protein_info:]
        x = residual + x

        # Bottleneck FFN
        x = self.protein_adapter_ffn(x)

        # ln and lm_head
        x = self.ln_f(x)
        lm_logits = self.lm_head(x)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
        )
