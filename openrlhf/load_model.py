from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer


config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", config=config, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo", torch_dtype=torch.bfloat16)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=32,
#     lora_alpha=16,
#     target_modules="all-linear",
#     lora_dropout=0,
#     bias="none",
# )

# model = get_peft_model(model, lora_config)