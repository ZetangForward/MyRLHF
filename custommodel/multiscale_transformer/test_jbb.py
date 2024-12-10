# transfromers version 4.38.2
# this example is tested with 4 RTX3090s, 24GB memory each
import warnings
warnings.filterwarnings("ignore")

import torch 
import json
import time
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from modifier import apply


window_size = 1024  
group_size = 32  # zecheng_note: 32 / 16
use_flash = True


def setup_model(args):
    model_name = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    config = AutoConfig.from_pretrained(model_name)
    config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, attn_implementation="flash_attention_2").half().cuda().eval()
    window_size=1024
    apply(model, args.group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attention_2")
    
    return model

def setup_tokenizer():
    return AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")


if __name__ == "__main__":
    model_name = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    config = AutoConfig.from_pretrained(model_name)
    config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, attn_implementation="flash_attention_2").half().cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attention_2")

    prompt_postfix = "What is the pass key? The pass key is "
    input_ids = tokenizer(prompt_postfix, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        tokens = model.generate(input_ids, do_sample=True, max_new_tokens=100, temperature=0.9)
        print(tokenizer.batch_decode(tokens))