from cd_llama import LoraModel, LoraConfig, LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained(
    "/data/hf_models/Meta-Llama-3.1-8B-Instruct", 
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
).cuda()


lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.01,
    bias="none",
)


lora_model = LoraModel(model, lora_config, "default")

# 伪造输入数据
batch_size = 2  # 批量大小
seq_length = 10  # 序列长度
vocab_size = model.config.vocab_size  # 词汇表大小

# 生成随机的 input_ids（假设词汇表大小为 vocab_size）
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()

# 生成 attention_mask（假设所有 token 都是有效的）
attention_mask = torch.ones_like(input_ids).cuda()

# 生成 labels（用于计算 loss，通常与 input_ids 相同）
labels = input_ids.clone()

# 将输入传递给模型
outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, cd_noise_settings={"add_noise": True})
# 获取 loss
loss = outputs.loss
print(f"Loss: {loss.item()}")


outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, cd_noise_settings={"add_noise": False})

# 获取 loss
loss = outputs.loss
print(f"Loss: {loss.item()}")

