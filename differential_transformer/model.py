from dataclasses import dataclass

import torch
import torch.distributed as dist
from flash_attn.layers.rotary import RotaryEmbedding
from einops import rearrange

from multihead_flashdiff_2 import MultiheadFlashDiff2
from flashdiff import FlashDiffAttention
from differential_transformer.rotary import apply_rotary_emb


@dataclass
class Args:
    model_parallel_size: int
    decoder_kv_attention_heads: int


def create_new_impl(origin_impl, head_dim, depth):
    diff_attn_func = FlashDiffAttention(
        head_dim=embed_dim // num_new_heads, depth=depth, causal=True
    ).to(device, dtype=dtype)
    # make the initialization the same
    diff_attn_func.lambda_q1.data.copy_(origin_impl.lambda_q1.data)
    diff_attn_func.lambda_k1.data.copy_(origin_impl.lambda_k1.data)
    diff_attn_func.lambda_q2.data.copy_(origin_impl.lambda_q2.data)
    diff_attn_func.lambda_k2.data.copy_(origin_impl.lambda_k2.data)
    #diff_attn_func.subln.weight.data.copy_(origin_impl.subln.weight.data)
    
    def new_impl(x, rel_pos):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = origin_impl.q_proj(x)
        k = origin_impl.k_proj(x)
        v = origin_impl.v_proj(x)

        # here we no longer need "// 2"
        num_heads = embed_dim // head_dim
        num_kv_heads = k.shape[-1] // head_dim

        q = q.view(bsz, tgt_len, num_heads, head_dim)
        k = k.view(bsz, src_len, num_kv_heads, head_dim)
        v = v.view(bsz, src_len, num_kv_heads, head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        output = diff_attn_func(q, k, v)
        output = rearrange(output, '... H D -> ... (H D)')

        output = origin_impl.out_proj(output)
        return output
    
    return new_impl


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    args = Args(model_parallel_size=1, decoder_kv_attention_heads=4)
    batch_size = 2
    num_heads = 16
    seq_len = 512
    embed_dim = 2048
    depth = 12
    # in the new implementation, the num_heads should be twice the original num_heads
    num_new_heads = num_heads * 2
    head_dim = embed_dim // num_new_heads

    print("initializing modules")
    origin_impl = MultiheadFlashDiff2(args, embed_dim=embed_dim, depth=depth, num_heads=num_heads).to(device, dtype=dtype)
    new_impl = create_new_impl(origin_impl, head_dim, depth)

    print("creating test data")
    rotary_emb = RotaryEmbedding(
        head_dim,
        base=10000.0,
        interleaved=True,
        device=device,
    )
    rotary_emb._update_cos_sin_cache(seq_len, device=device, dtype=torch.bfloat16)
    rel_pos = (rotary_emb._cos_cached, rotary_emb._sin_cached)
    hidden_states = torch.randn((batch_size, seq_len, embed_dim), device=device, dtype=dtype)

    print("run origin forward")
    origin_output = origin_impl(hidden_states, rel_pos)

    print("run new forward")
    new_output = new_impl(hidden_states, rel_pos)

    assert torch.allclose(origin_output, new_output, atol=1e-6)