from modelzipper.tutils import *
import sys
import os
from transformers import LlamaForCausalLM
sys.path.append("/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect")
from retrieval_head_detection import SentenceSampler
import datasets
from baukit import Trace
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model = LlamaForCausalLM.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct", device_map='balanced_low_0', torch_dtype=torch.bfloat16, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
data = auto_read_data("/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle.jsonl")

# needle
needles_and_stacks = [json.loads(l) for l in open(f"/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle.jsonl")]
needle_list = [l["needle"] for l in needles_and_stacks]
retrieval_question_list = [l["question"] for l in needles_and_stacks]
evidence_list = [l["real_needle"] for l in needles_and_stacks]
golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
tags = [l["tag"] for l in needles_and_stacks]

selected_idx = 0
needle = [tokenizer(i)['input_ids'] for i in needle_list[selected_idx]]
evidence = [tokenizer(i)['input_ids'] for i in evidence_list[selected_idx]]
question = retrieval_question_list[selected_idx]
answer = golden_answer_list[selected_idx]
tag = tags[selected_idx]

# 初始化采样器
haystack = datasets.load_dataset("/data/data/zecheng/data/pg19-test", split="test")
noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=None)
background_text = noise_sampler_test.get_sample(15500)
disturb_tok_needles = [i for i in needle if i not in evidence]
disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))
depth_percent = [0.1, 0.2, 0.4]

updated_sample = [[] for _ in range(len(background_text) + 1)]
real_pos = [int(len(background_text) * i) for i in depth_percent]
for fact, pos in zip(evidence, real_pos):  # insert real needle
    updated_sample[pos].append(fact)
for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
    updated_sample[pos].append(fact)
for i, s in enumerate(background_text):  # insert irrevelent needle
    updated_sample[i].append(s)

flat = [i for s in updated_sample for i in s]
tokens = [i for s in flat for i in s]

new_context = tokenizer.decode(tokens)
input_context = new_context + f"\nQuestion: {question}\nAnswer:"
inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, return_tensors='pt').to(model.device)

print(inp.shape)

with torch.no_grad(), Trace(model.model.layers[8], "self_attn") as ret:
    _ = model(inp, output_attentions=True)
    representation = ret.output