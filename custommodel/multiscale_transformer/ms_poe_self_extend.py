from typing import Optional, Tuple

import os
import sys
import pdb
import math
import copy
import time 
import types
import numpy as np 
from scipy.stats import entropy

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
     LlamaFlashAttention2,
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaForCausalLM,
)
from flash_attn import flash_attn_func
from selfextend_flash_attn import self_extend_flash_forward


__all__ = ['MsPoELlamaForCausalLM']


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):   
    cos = cos[:, position_ids] # [head, seq_len, dim] -> [head, bs, seq_len, dim]
    sin = sin[:, position_ids] # [head, seq_len, dim] -> [head, bs, seq_len, dim]
    cos = cos.transpose(0, 1)  # [head, bs, seq_len, dim] -> [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [head, bs, seq_len, dim] -> [bs, head, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed
    

def sample_rotary_emb(cos, sin, num_key_value_groups):
    cos = cos[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    sin = sin[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    return cos, sin


### Positional Scaling
class MsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads,1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)

        t = t / compress_ratio
        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:,:seq_len].to(dtype=x.dtype),
            self.sin_cached[:,:seq_len].to(dtype=x.dtype),
        )
 

def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) # only consider bsz=1 for now.



    neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position)#, seq_len=None)
    neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position)#, seq_len=None)


    _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 // group_size_1
    group_key_position = key_position // group_size_1

    group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position)#, seq_len=None)
    group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position)#, seq_len=None)



    neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
    _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
    group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
    _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)



    neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
    group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)



    neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 


    if attention_mask is not None:  # no matter the length, we just slice it
        if cache_position is not None:
            causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
        else:
            causal_mask = attention_mask
        group_attn_weights = group_attn_weights + causal_mask
        neighbor_attn_weights = neighbor_attn_weights + causal_mask

    if q_len == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -group_size_2:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len-group_size_2 > 0:
            group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
            neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    neighbor_attention_mask = neighbor_attention_mask.bool()
    attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def flash_self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    ========================= forward logic =========================
        Require updating tansformers to >= 4.38.2, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """
    
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) 
    attn_dropout = self.config.attention_dropout if self.training else 0.0

    if q_len == 1:
        # We implement the case q_len == 1 separately, by manipulating positions.
        # for our flash implementation doesnot work for  decoding stage at the releasing time.

        neighbor_key_position = position_ids[:, -1] - key_position
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_key_position = position_ids[:, -1]//group_size_1 - key_position//group_size_1 + (_re_group_size_2 - _re_group_size_2//group_size_1)
        decode_key_position = torch.cat([group_key_position[:, :-group_size_2], neighbor_key_position[:,-group_size_2:]], dim=1)
        
        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position)
        # neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position_ids) 
        decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
        _, decode_key_states = apply_rotary_pos_emb(scaled_query, key_states, decode_k_cos, -decode_k_sin, decode_key_position) 

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        
        attn_output = flash_attn_func(
            decode_query_states,
            decode_key_states,
            decode_value_states,
            dropout_p=attn_dropout, 
            softmax_scale=None, 
            causal=True
        )

    elif q_len == kv_seq_len:
        # set correct position_ids & apply RoPE.
        neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position)
        # neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position)

        _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position)
        # group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position)

        # _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
        # _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)
        neighbor_query_states, neighbor_key_states = apply_rotary_pos_emb(scaled_query, key_states, neighbor_q_cos, neighbor_q_sin)
        group_query_states, group_key_states = apply_rotary_pos_emb(scaled_query, key_states, group_q_cos, group_q_sin)
        
        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        
        attn_output = self_extend_flash_forward(self,
                                                query_position,
                                                group_size_2,
                                                neighbor_query_states,
                                                neighbor_key_states,
                                                group_query_states,
                                                group_key_states,
                                                value_states,
                                                attention_mask,
                                                bsz,
                                                q_len,
                                                kv_seq_len,
                                                attn_dropout,
                                            )
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


   


### Layer-wise Positional Scaling
class LayerWiseMsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, layer_idx=0, num_heads=32, num_layers=32, max_position_embeddings=2048, base=10000, device=None, chunk_size=4096, layer_wise=False, head_wise=True):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.layer_wise = layer_wise 
        self.head_wise = head_wise   

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            layer_idx=layer_idx, seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, layer_idx, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads,1)
        layer_compress_ratio = 1.0
        head_compress_ratio = 1.0
        if self.layer_wise:
            layer_compress_ratio = (layer_idx + 1) / self.num_layers
        if self.head_wise:
            head_compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
            head_compress_ratio = min_ratio + (max_ratio - min_ratio) * (head_compress_ratio / num_heads)
            head_compress_ratio = 1 / head_compress_ratio.unsqueeze(-1)

        t = t * layer_compress_ratio * head_compress_ratio # zecheng_note: compress_ratio 如果越大，token之间的距离越大，token之间的角度差越小，相反，如果compresion ratio越大，词之间的距离越大，token之间的角度差越大

        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:,:seq_len].to(dtype=x.dtype),
            self.sin_cached[:,:seq_len].to(dtype=x.dtype),
        )



class MsPoELlamaFlashAttention(LlamaFlashAttention2):

    def __init__(self, layer_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_ratio_min = self.config.compress_ratio_min
        self.compress_ratio_max = self.config.compress_ratio_max
        self.head_order = None
        self.enable_head_metrics = False
        self.layer_idx = layer_idx

        self.rotary_emb = LayerWiseMsPoELlamaRotaryEmbedding(
            self.head_dim,
            min_cratio=self.compress_ratio_min,
            max_cratio=self.compress_ratio_max,
            layer_idx=layer_idx,
            num_layers = self.config.num_hidden_layers,
            num_heads=self.num_heads,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            layer_wise=True,
            head_wise=False,
        )

    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).float().mean(-1)[:,-1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders
    

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):

        query_states_new = query_states
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders


    def forward(self, hidden_states, attention_mask = None, position_ids = None, past_key_value = None, output_attentions = False, use_cache = False, cache_position = None, position_embeddings = None, **kwargs):
        
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        
        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz, attention_mask)
            self.enable_head_metrics = False

        if self.head_order:  # avoid None to add a new dimension
            head_cos = cos[self.head_order, :, :]
            head_sin = sin[self.head_order, :, :]
        else:
            head_cos, head_sin = cos, sin

        query_states = apply_rotary_pos_emb_single_scaling(query_states, head_cos, head_sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MsPoELlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):

        query_states_new = query_states
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders


    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).float().mean(-1)[:,-1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MsPoELlamaRotaryEmbedding(
                self.head_dim,
                min_cratio=self.compress_ratio_min,
                max_cratio=self.compress_ratio_max,
                num_heads=self.num_heads,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                assert False # not implemented
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    min_cratio=self.compress_ratio_min,
                    max_cratio=self.compress_ratio_max,
                    num_heads=self.num_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                assert False # not implemented
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz, attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]
        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # key/value are already rotated
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MsPoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in range(config.num_hidden_layers):
                self.model.layers[layer_idx].self_attn = MsPoELlamaFlashAttention(layer_idx, config)
                # self.model.layers[layer_idx].self_attn = LlamaFlashAttention2(config)

    def _reset(self):
        for layer_idx in range(self.config.num_hidden_layers):
            self.model.layers[layer_idx].self_attn.enable_head_metrics = False
            self.model.layers[layer_idx].self_attn.head_order = None


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    import datasets
    config = AutoConfig.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
    config.compress_ratio_min = 1.2
    config.compress_ratio_max = 1.8
    model = MsPoELlamaForCausalLM.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16,config=config).cuda()
    test_set = datasets.load_dataset("/data/data/zecheng/data/pg19-test", split="test")
    one_sample = test_set[0]['text']
    tokenizer = AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
    # input_ids = tokenizer(one_sample, return_tensors="pt")["input_ids"].to(model.device)
    user_query = "hello"
    input_ids = tokenizer(user_query, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.9)
        print(tokenizer.batch_decode(output, skip_special_tokens=True))
