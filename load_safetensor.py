from safetensors import safe_open

ckpt_path = "/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step150/adapter_model.safetensors"

tensors = {}
with safe_open(ckpt_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

import ipdb; ipdb.set_trace()