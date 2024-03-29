# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    gqa_aware: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])

transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(block_size=32768, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim,
                 bifurcated_attn, context_seq_len, max_new_tokens, num_parallel_samples,
                 dtype=torch.bfloat16):
        super().__init__()
        self.bifurcated_attn = bifurcated_attn
        self.context_seq_len = context_seq_len
        if self.bifurcated_attn:
            context_cache_shape = (1, n_heads, context_seq_len, head_dim)
            dec_cache_shape = (num_parallel_samples, n_heads, max_new_tokens, head_dim)
            self.register_buffer('k_cache_context', torch.zeros(context_cache_shape, dtype=dtype))
            self.register_buffer('v_cache_context', torch.zeros(context_cache_shape, dtype=dtype))
            self.register_buffer('k_cache_dec', torch.zeros(dec_cache_shape, dtype=dtype))
            self.register_buffer('v_cache_dec', torch.zeros(dec_cache_shape, dtype=dtype))
            self.k_cache = {'context': self.k_cache_context, 'dec': self.k_cache_dec}
            self.v_cache = {'context': self.v_cache_context, 'dec': self.v_cache_dec}
        else:
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
            self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
            self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val, prefill=False):
        # print("Updating. bifurcated attn", self.bifurcated_attn)
        if self.bifurcated_attn:
            if prefill:
                assert self.k_cache['context'].shape == k_val.shape
                self.k_cache['context'][:] = k_val
                self.v_cache['context'][:] = v_val
                return {'context': self.k_cache['context'], 'dec': None}, \
                    {'context': self.v_cache['context'], 'dec': None}
            else:
                assert len(input_pos) == 1, f"input_pos: {input_pos} should be of length 1 during decoding"
                assert k_val.size(-2) == 1
                new_input_pos = input_pos - self.context_seq_len
                self.k_cache['dec'][:, :, new_input_pos] = k_val
                self.v_cache['dec'][:, :, new_input_pos] = v_val
                return self.k_cache, self.v_cache

        else:
            # input_pos: [S], k_val: [B, H, S, D]
            assert input_pos.shape[0] == k_val.shape[2]

            k_out = self.k_cache
            v_out = self.v_cache

            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

            return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, hard_reset=False,
                     bifurcated_attn=False, context_seq_len=0, max_new_tokens=0,
                     parallel_samples=1):
        self.bifurcated_attn = bifurcated_attn
        if (not hard_reset) and self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        # print("Initial max sequence length", max_seq_length) # 4096
        max_seq_length = find_multiple(max_seq_length, 8)
        # print("Setting max sequence length to be ", max_seq_length) $ 4200 (4096 + 100  and round to multiples of 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim,
                                           bifurcated_attn, context_seq_len, max_new_tokens, parallel_samples)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, prefill: bool=False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos] # [1,1, 8192, 8196]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask, prefill=prefill)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str, gqa_aware=False, block_size=None):
        config = ModelArgs.from_name(name)
        config.gqa_aware = gqa_aware
        if block_size is not None:
            config.block_size = block_size
        return cls(config)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, prefill: bool) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos, prefill=prefill)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.gqa_aware = config.gqa_aware

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, prefill=False) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v, prefill=prefill)
            if not self.kv_cache.bifurcated_attn:
                k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            else:
                if prefill:
                    k,v = k['context'], v['context']
                    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:,:,:,:len(input_pos)], dropout_p=0.0)
                else:
                    k_context, k_dec = k['context'], k['dec']
                    v_context, v_dec = v['context'], v['dec']
                    
                    if not self.gqa_aware:
                        k_context = k_context.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        v_context = v_context.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        k_dec = k_dec.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        v_dec = v_dec.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

                    mock_bifurcated = False 
                    if not mock_bifurcated:
                        # for testing purposes
                        if self.gqa_aware:
                            y = scaled_dot_product_attention_bifurcated_gqa(
                                q, k_context, k_dec,
                                v_context, v_dec,
                                attn_mask=None,
                                dropout_p=0.0
                            )
                        else:
                            y = scaled_dot_product_attention_bifurcated(q, k_context, k_dec,
                                                                    v_context, v_dec,
                                                                    attn_mask=None,
                                                                    dropout_p=0.0
                                                                    )
                    else:
                        k_context = k_context.repeat(q.shape[0], 1, 1, 1)
                        v_context = v_context.repeat(q.shape[0], 1, 1, 1)
                        k = torch.cat([k_context, k_dec], dim=-2)
                        v = torch.cat([v_context, v_dec], dim=-2)
                        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:,:,:,:k.size(-2)], dropout_p=0.0)
        
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # print('at rotary - x shape', x.shape)
    # print('freq_cis', freqs_cis.shape)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    # attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    if attn_mask is not None:
        print("attn_mask", attn_mask)
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
    else:
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor), dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ V

def scaled_dot_product_attention_einsum(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # q: bhnk, k: bgmk, v: bgmv
    # or       k: gmk   v: gmv
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    if attn_mask is not None:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((torch.einsum("bhnk,bhmk->bhnm", Q, K) * scale_factor) + attn_mask, dim=-1)
    else:
        attn_weight = torch.softmax((torch.einsum("bhnk,bhmk->bhnm", Q, K) * scale_factor), dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return torch.einsum("bhnm,bhmv->bhnv", attn_weight, V)

def scaled_dot_product_attention_bifurcated_context(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # q: bhnk, 
    #          k: 1gmk   v: 1gmv
    assert K.size(0) == 1 and V.size(0) == 1, "K should have a batch size of 1 for bifurcated context"
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    if attn_mask is not None:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((torch.einsum("bhnk,hmk->bhnm", Q, K) * scale_factor) + attn_mask, dim=-1)
    else:
        attn_weight = torch.softmax((torch.einsum("bhnk,hmk->bhnm", Q, K) * scale_factor), dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return torch.einsum("bhnm,hmv->bhnv", attn_weight, V)

def scaled_dot_product_attention_bifurcated_incremental(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    assert Q.size(-2) == 1, "Q should have a sequence length of 1 for incremental decoding"
    # note: generalize for speculative decoding
    return scaled_dot_product_attention_einsum(Q, K, V, attn_mask, dropout_p, is_causal, scale)

def scaled_dot_product_attention_bifurcated(Q, K_context, K_dec, V_context, V_dec, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # q: bhnk, 
    # context     k: 1hMk   v: 1hMv
    # incremental k: bhmk   v: bhmv

    assert K_context.size(0) == 1 and V_context.size(0) == 1, "K should have a batch size of 1 for bifurcated context"
    assert Q.size(-2) == 1, f"At incremental decoding phase with bifurcated attention, expecting query length = 1. Current query length = {Q.size(-2)}"
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    attn_weight = torch.softmax(
            (torch.cat([torch.einsum("bhnk,hMk->bhnM", Q, K_context.squeeze(0)),
                        torch.einsum("bhnk,bhmk->bhnm", Q, K_dec)
                        ], dim=-1) * scale_factor),
            dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    M = K_context.size(-2)
    attn_weight_context, attn_weight_dec = attn_weight[:,:,:,:M], attn_weight[:,:,:,M:]
    return torch.einsum("bhnM,hMv->bhnv", attn_weight_context, V_context.squeeze(0)) + \
        torch.einsum("bhnm,bhmv->bhnv", attn_weight_dec, V_dec)


def scaled_dot_product_attention_bifurcated_gqa(Q, K_context, K_dec, V_context, V_dec, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # q: bhnk, 
    # context     k: 1gMk   v: 1gMv
    # incremental k: bgmk   v: bgmv

    assert K_context.size(0) == 1 and V_context.size(0) == 1, "K should have a batch size of 1 for bifurcated context"
    assert Q.size(-2) == 1, f"At incremental decoding phase with bifurcated attention, expecting query length = 1. Current query length = {Q.size(-2)}"
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    # reshape Q to be bpgnk where h = p*g numerically
    assert K_context.size(1) == V_context.size(1)
    g = K_context.size(1)
    h = Q.size(1)
    assert h % g == 0
    Q = Q.view(Q.size(0), -1, g, Q.size(2), Q.size(3))


    attn_weight = torch.softmax(
            (torch.cat([torch.einsum("bpgnk,gMk->bpgnM", Q, K_context.squeeze(0)),
                        torch.einsum("bpgnk,bgmk->bpgnm", Q, K_dec)
                        ], dim=-1) * scale_factor),
            dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    M = K_context.size(-2)
    attn_weight_context, attn_weight_dec = attn_weight[:,:,:,:,:M], attn_weight[:,:,:,:,M:]
    y = torch.einsum("bpgnM,gMv->bpgnv", attn_weight_context, V_context.squeeze(0)) + \
        torch.einsum("bpgnm,bgmv->bpgnv", attn_weight_dec, V_dec)
    y = y.reshape(y.size(0), h, y.size(-2), y.size(-1))
    return y