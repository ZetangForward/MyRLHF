from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftModelForCausalLM, PeftConfig
import torch

# 配置路径
# model_path = "Qwen/Qwen2.5-7B-Instruct"  # 替换为模型的路径或名称
# adapter_path = "/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Qwen2.5-7B-Instruct/simpo/global_step50"  # 替换为 PEFT adapter 的路径
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step125"
save_path = "./"  # 替换为保存合并后模型的路径

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2="flash_attention_2",
    trust_remote_code=True
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)

# 如果有 adapter，则加载并合并
if adapter_path:
    # 加载 PEFT adapter
    model = PeftModelForCausalLM.from_pretrained(model, adapter_path, ignore_mismatched_sizes=False)
    print("Loaded PEFT adapter from:", adapter_path)
    
    # 合并权重
    print("Merging model and PEFT adapter weights...")
    model = model.merge_and_unload()
import ipdb; ipdb.set_trace()
# 保存合并后的模型
print(f"Saving merged model to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model saved successfully!")
