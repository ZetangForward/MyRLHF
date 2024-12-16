
from typing import Optional, Tuple

import os, sys, pdb, math, time, types
import numpy as np 

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func,flash_attn_qkvpacked_func 

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaAttention,
    repeat_kv,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from feedforward import CoPEMLP
from normalize import AddNorm, PreNorm


def get_mask(lens, valid_len, zero_triu, true_is_keep=False):
    '''
    l        : int or tuple  (q len,k len)
    valid_len: (b, ) (k len)
    '''
    q_len,k_len=(1, lens) if isinstance(lens, int) else lens

    if valid_len is None:
        return None
    if valid_len.ndim>2:# means valid_len is actually a mask not len 
        return valid_len
    mask=(torch.arange(k_len).to(valid_len.device)[None,:]<valid_len[:,None]) # (b, kl)
    mask=mask[:,None,:].repeat(1,q_len,1) # (b, ql, kl)
    if zero_triu: # decoder self attention
        mask=torch.tril(mask)
    mask=mask[:,None,:,:] # (b, 1(h), ql, kl)
    return mask if true_is_keep else ~mask


class ContextualPositionEmbedding(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max=npos_max
        self.pos_emb=nn.Parameter(torch.zeros(1, 1, head_dim, npos_max))

    def forward(self, query, attn_logits, mask = None):
        #query: (batch_size, num_heads, q_len or 1, ndims)
        #attn_logits : (batch_size, num_heads, q_len or 1, k_len)
        #mask:  (b, 1, q_len, k_len) , None means evaluating and using kv cache
        if mask is not None:
            attn_logits += mask
        gates=torch.sigmoid(attn_logits)
        pos=gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos=pos.clamp(max=self.npos_max-1)  # (b, h, q_len, k_len)

        #interpolate from integar positions
        pos_ceil   = pos.ceil().long()
        pos_floor  = pos.floor().long() # (b, h, q_len, k_len)
        logits_int = torch.matmul(query, self.pos_emb) #q(e^T) :(b, h, q_len, npos_max) 
        logits_ceil =logits_int.gather(dim=-1, index=pos_ceil)
        logits_floor=logits_int.gather(dim=-1, index=pos_floor)
        w = pos - pos_floor

        attn_logits += logits_ceil * w + logits_floor * (1-w)

        return attn_logits
    
class CoPELlamaAttention(nn.Module):
    def __init__(self,
                 config,
                 layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.sqrt_head_dim = self.head_dim**0.5
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.cope_enc=ContextualPositionEmbedding(npos_max=config.npos_max, head_dim=self.head_dim)

    def forward_qkv(self,q,k,v):
        b,_,_=q.shape
        Q=self.q_proj(q).view(b,q.size(1),-1,self.head_dim).transpose(1,2)
        K=self.k_proj(k).view(b,k.size(1),-1,self.head_dim).transpose(1,2)
        V=self.v_proj(v).view(b,v.size(1),-1,self.head_dim).transpose(1,2)
        return Q,K,V
    
    def forward_attention(self,QK,V,mask=None):
        if type(QK) is tuple:
            Q,K=QK
            logits = torch.matmul(Q,K.transpose(-2,-1))/self.sqrt_head_dim           
        else: logits = QK
        
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(mask, neg_inf)

        score = torch.softmax(logits, dim=-1, dtype=torch.float32)
        if mask is not None:
            score = score.masked_fill(mask, 0.0)
        
        score = F.dropout(score, p=self.attention_dropout, training=self.training)

        out = torch.matmul(score, V)
        out = out.transpose(1,2).contiguous().reshape(V.size(0), -1, self.hidden_size)
        
        return out, score

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert attention_mask.ndim==4,"mask shape: " + str(attention_mask.shape) + "is not allowed!"
        
        bsz, q_len, _ =hidden_states.size()
        
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

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        else:
            query_states, key_states, value_states = self.forward_qkv(hidden_states, hidden_states, hidden_states)# (bs, h_len, nheads, ndim)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / self.sqrt_head_dim
        
        bool_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            bool_mask = causal_mask!=0.
            attn_weights = attn_weights + causal_mask

        attn_weights = self.cope_enc(query_states, attn_weights)

    

        attn_output, attn_weights = self.forward_attention(attn_weights, value_states, bool_mask)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
         





class CoPELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers=len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn=CoPELlamaAttention(config, layer_idx)

if __name__=="__main__":
    from transformers import AutoConfig, AutoTokenizer
    import datasets
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    config = AutoConfig.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
    config.npos_max = 64
    config._attn_implementation="eager"
    test_set = datasets.load_dataset("/data/data/zecheng/data/pg19-test", split="test")
    one_sample = test_set[0]['text']
    tokenizer = AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
    input_ids=tokenizer(one_sample,return_tensors='pt')
    model = CoPELlamaForCausalLM(config).cuda()
    attention_mask=input_ids['attention_mask']
    input_ids=input_ids['input_ids'][:,:20].cuda()

    model_output = model(input_ids=input_ids[:,:-1],labels=input_ids[:,1:])

    print(model_output.loss)
