import torch
import math
from torch import nn
import sys
from typing import List, Optional, Tuple, Union
import importlib
from modelzipper.tutils import *
from transformers.cache_utils import StaticCache
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from flash_attn import flash_attn_func
sys.path.append('/data/zecheng/acl2025/MyRLHF/differential_transformer')
from rotary import apply_rotary_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LlamaDiffAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lambda_init = lambda_init_fn(self.layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads // 2, 2, self.head_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        rel_pos = (cos, sin)
        query_states = apply_rotary_emb(query_states, *rel_pos, interleave=True)
        key_states = apply_rotary_emb(key_states, *rel_pos, interleave=True)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.reshape(bsz, q_len, self.num_heads // 2, 2, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads // 2, 2, self.head_dim)

        q1, q2 = query_states[:, :, :, 0], query_states[:, :, :, 1]
        k1, k2 = key_states[:, :, :, 0], key_states[:, :, :, 1]
        v1, v2 = value_states[:, :, :, 0], value_states[:, :, :, 1]
        
        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_output = attn1 - lambda_full * attn2

        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



# 动态导入 transformers.modeling_llama
modeling_llama = importlib.import_module("transformers.models.llama.modeling_llama")

# 修改 LLAMA_ATTENTION_CLASSES 字典，插入自定义 Attention 类
modeling_llama.LLAMA_ATTENTION_CLASSES["diff_attn"] = LlamaDiffAttention

config = LlamaConfig.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
config.attention_type = "diff_attn"

model = LlamaForCausalLM.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct", config=config, torch_dtype=torch.bfloat16).to('cuda:7')
tokenizer = AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")

data = auto_read_data("/data/zecheng/Long-form-reasoning-data/data/generated_tasks/qa3/64k.json")
test_case = data[0]['input']

input_ids = tokenizer(test_case[:16000], return_tensors="pt").to('cuda:7')
print(input_ids.input_ids.shape)
loss = model(**input_ids)