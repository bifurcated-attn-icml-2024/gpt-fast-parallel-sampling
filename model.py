# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math
import os

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
                 bifurcated_kv, bifurcated_attn, context_seq_len, max_new_tokens, num_parallel_samples,
                use_flash2_prefill=False, use_flash2_decode=False, 
                use_sdpa_flash=False,
                 dtype=torch.bfloat16):
        super().__init__()
        self.bifurcated_kv = bifurcated_kv
        self.bifurcated_attn = bifurcated_attn
        self.context_seq_len = context_seq_len
        self.use_flash2_prefill = use_flash2_prefill
        self.use_flash2_decode = use_flash2_decode
        self.use_sdpa_flash = use_sdpa_flash
        if self.bifurcated_kv:
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
        if self.bifurcated_kv:
            if prefill:
                assert self.k_cache['context'].shape == k_val.shape
                self.k_cache['context'][:] = k_val
                self.v_cache['context'][:] = v_val
                return {'context': self.k_cache['context'], 'dec': None}, \
                    {'context': self.v_cache['context'], 'dec': None}
            else:
                # incremental decoding
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
                     bifurcated_kv=False, bifurcated_attn=False,
                     context_seq_len=0, max_new_tokens=0,
                     parallel_samples=1, use_flash2_prefill=False, use_flash2_decode=False,
                     use_sdpa_flash=False,
                     ):
        
        self.bifurcated_kv = bifurcated_kv
        self.bifurcated_attn = bifurcated_attn

        if (not hard_reset) and self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim,
                                           bifurcated_kv, bifurcated_attn, context_seq_len, max_new_tokens, parallel_samples,
                                           use_flash2_prefill, use_flash2_decode, use_sdpa_flash
                                           )

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, prefill: bool=False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # causal_mask: [1,1, q_len, k_len]
        mask = self.causal_mask[None, None, input_pos] # a slice along query length
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

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, prefill=False,
                ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        """
            Note: this transpose changes the dim from `bmgk` to `bgmk`
            FlashAttention2 expects the shape to be `bmhk` where `m` is the sequence length however
        """
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            # for incremental decoding, update kv cache with kv for new tokens
            k, v = self.kv_cache.update(input_pos, k, v, prefill=prefill)
        
        use_flash2_prefill = self.kv_cache.use_flash2_prefill
        use_flash2_decode = self.kv_cache.use_flash2_decode

        if prefill:
            if self.kv_cache.bifurcated_kv:
                k,v = k['context'], v['context']
            # k = k[:,:,:q.size(-2)]
            # v = v[:,:,:q.size(-2)]

            if use_flash2_prefill:
                if not self.gqa_aware:
                    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                # q: bhmk, k: bhmk, v: bhmk -> bnhv
                q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
                y = flash_attn_func(
                    q,
                    k,
                    v,
                    0.0,
                    softmax_scale=1/math.sqrt(self.head_dim),
                    causal=True,
                ).transpose(1, 2) # bnhv -> bhnv
            else:
                k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                # use is_causal=True instead of attn_mask=mask since it is compatible with SDPA flash kernel
                # and Math SDPA kernel (and much faster as well)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

                # note: above is functionally equivalent to
                # y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
                # or 
                # y = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, dropout_p=0.0)
                # or 
                # y = scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0)
                # or 
                # y = scaled_dot_product_attention_einsum(q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0)
        else:
            # incremental decoding
            if self.kv_cache.bifurcated_kv:
                # decode with bifurcated attention
                k_context, k_dec = k['context'], k['dec']
                v_context, v_dec = v['context'], v['dec']
                if self.kv_cache.bifurcated_attn:
                    if self.gqa_aware:
                        y = scaled_dot_product_attention_bifurcated_gqa(
                            q, k_context, k_dec,
                            v_context, v_dec,
                            attn_mask=mask, # compatible with compile
                            # input_pos=input_pos[0], # not compatible with compile
                        )
                    else:
                        k_context = k_context.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        v_context = v_context.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        k_dec = k_dec.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        v_dec = v_dec.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        y = scaled_dot_product_attention_bifurcated(
                            q, k_context, k_dec,
                            v_context, v_dec,
                            attn_mask=mask,
                        )
                else:
                    # join K and V together (use .expand otherwise torch.repeat consume too much memory)
                    k_context = k_context.expand(q.shape[0], -1, -1, -1)
                    v_context = v_context.expand(q.shape[0], -1, -1, -1)
                    k = torch.cat([k_context, k_dec], dim=-2) # bhmk or bgmk | m is at dim = -2
                    v = torch.cat([v_context, v_dec], dim=-2)

                    if not self.gqa_aware:
                        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                    
                    if use_flash2_decode:
                        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
                        y = flash_attn_func(
                            q,
                            k[:,:input_pos[0]+1,:,:],
                            v[:,:input_pos[0]+1,:,:],
                            0.0,
                            softmax_scale=None,
                            causal=False,
                        ).transpose(1, 2)
                    else:
                        if self.kv_cache.use_sdpa_flash:
                            k = k[:, :, :input_pos[0]+1]
                            v = v[:, :, :input_pos[0]+1]
                            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
                        else:
                            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
                        

            elif use_flash2_decode:
                if not self.gqa_aware:
                    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
                # using `:input_pos[0]+1` to avoid attending to future tokens
                # since we do not use explicit mask
                y = flash_attn_func(
                    q,
                    k[:,:input_pos[0]+1,:,:],
                    v[:,:input_pos[0]+1,:,:],
                    0.0,
                    softmax_scale=None,
                    causal=False, # true or false yileds the same result for query length = 1
                ).transpose(1, 2)
            else:
                k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

                if self.kv_cache.use_sdpa_flash:
                    # compatible with both flash and sdpa
                    k = k[:,:,:input_pos[0]+1,:]
                    v = v[:,:,:input_pos[0]+1,:]
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
                    # note: is_causal=True cannot be used with flash SDPA
                    #   Error: flash attention does not support the is_causal flag when seqlen_q != seqlen_k. Got seqlen_q:
                    #   1 seqlen_k: 8320. If you would like to use causal attention with non-square masks, please see CausalAttnMask.
                    #   y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                else:
                    # below is not compatible with flash SDPA due to attn_mask=mask
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
                

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


# tested and is correct compared to SDPA
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    if is_causal:
        assert attn_mask is None
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.size(), dtype=query.dtype, device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ value


# same as `scaled_dot_product_attention` but uses einsum
def scaled_dot_product_attention_einsum(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # q: bhnk, k: bhmk   v: bhmv
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if is_causal:
        assert attn_mask is None
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.size(), dtype=query.dtype, device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = torch.einsum("bhnk,bhmk->bhnm", query, key) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return torch.einsum("bhnm,bhmv->bhnv", attn_weight, value)


def scaled_dot_product_attention_bifurcated(Q, K_context, K_dec, V_context, V_dec, input_pos=None, attn_mask=None, dropout_p=0.0, scale=None):
    # q: bhnk, 
    # context     k: 1hMk   v: 1hMv
    # incremental k: bhmk   v: bhmv
    assert K_context.size(0) == 1 and V_context.size(0) == 1, "K should have a batch size of 1 for bifurcated context"
    assert Q.size(-2) == 1, f"At incremental decoding phase with bifurcated attention, expecting query length = 1. Current query length = {Q.size(-2)}"
    
    if attn_mask is not None:
        attn_bias = torch.zeros(*attn_mask.shape, dtype=Q.dtype, device=Q.device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
        slice_pos_dec = input_pos - K_context.size(-2) + 1
        K_dec, V_dec = K_dec[:,:,:,slice_pos_dec:], V_dec[:,:,:,slice_pos_dec:]
        attn_bias = 0
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    attn_weight = torch.softmax(
            (torch.cat([torch.einsum("bhnk,hMk->bhnM", Q, K_context.squeeze(0)),
                        torch.einsum("bhnk,bhmk->bhnm", Q, K_dec)
                        ], dim=-1) * scale_factor) + attn_bias,
            dim=-1)
    
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    M = K_context.size(-2)
    attn_weight_context, attn_weight_dec = attn_weight[:,:,:,:M], attn_weight[:,:,:,M:]
    return torch.einsum("bhnM,hMv->bhnv", attn_weight_context, V_context.squeeze(0)) + \
        torch.einsum("bhnm,bhmv->bhnv", attn_weight_dec, V_dec)


def scaled_dot_product_attention_bifurcated_gqa(Q, K_context, K_dec, V_context, V_dec, input_pos=None, attn_mask=None, dropout_p=0.0, scale=None):
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
    Q = Q.view(Q.size(0),  g, -1, Q.size(2), Q.size(3))

    if attn_mask is not None:
        attn_bias = torch.zeros(*attn_mask.shape, dtype=Q.dtype, device=Q.device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
        slice_pos_dec = input_pos - K_context.size(-2) + 1
        K_dec, V_dec = K_dec[:,:,:,slice_pos_dec:], V_dec[:,:,:,slice_pos_dec:]
        attn_bias = 0
    attn_weight = torch.softmax(
            (torch.cat([torch.einsum("bgpnk,gMk->bgpnM", Q, K_context.squeeze(0)),
                        torch.einsum("bgpnk,bgmk->bgpnm", Q, K_dec)
                        ], dim=-1) * scale_factor) + attn_bias,
            dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    M = K_context.size(-2)
    attn_weight_context, attn_weight_dec = attn_weight[:,:,:,:,:M], attn_weight[:,:,:,:,M:]
    y = torch.einsum("bgpnM,gMv->bgpnv", attn_weight_context, V_context.squeeze(0)) + \
        torch.einsum("bgpnm,bgmv->bgpnv", attn_weight_dec, V_dec)
    y = y.reshape(y.size(0), h, y.size(-2), y.size(-1))
    return y