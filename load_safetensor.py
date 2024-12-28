from safetensors import safe_open

ckpt_path = "/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step150/adapter_model.safetensors"

tensors = {}
with safe_open(ckpt_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

import ipdb; ipdb.set_trace()



"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("llama3.1")

## tokenizer 0-shot
tok = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
        # {
        #     "role": "assistant",
        #     "content": "Paris",
        # },
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)


## tokenizer 1-shot, demonstration (错)
demonstration = "Q: 1 + 1 = ?\n A: 2\n"
tok = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": f"{demonstration}\n2 + 2 = ?",
        },
        # {
        #     "role": "assistant",
        #     "content": "Paris",
        # },
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)

## tokenizer 1-shot, demonstration (错)
demonstration = "Q: 1 + 1 = ?\n A: 2\n"
tok = tokenizer.apply_chat_template(
    [   
        {
            "role": "system",
            "content": "xxx",
        },
        {
            "role": "user",
            "content": "Q: 1 + 1 = ?",
        },
        {
            "role": "assistant",
            "content": "2",
        },
        {
            "role": "user",
            "content": "Q: 1 + 1 = ?",
        },
        {
            "role": "assistant",
            "content": "2",
        },
        {
            "role": "user",
            "content": "2 + 2 = ?",
        },
        # {
        #     "role": "assistant",
        #     "content": "Paris",
        # },
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)

"""