#import tiktoken
import os 
import glob
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faiss_attn'))
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
from source.modeling_phi3 import Phi3ForCausalLM
import numpy as np
import argparse
import datasets
import itertools
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import time
import nltk
import torch
import random
from tqdm import tqdm, trange
from loguru import logger

class SentenceSampler:
    def __init__(self, dataset, tokenizer, min_sentence_len=10, max_sentence_len=None, shuffle=False, random_seed=42):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.sentence_tokenizer = nltk.PunktSentenceTokenizer()
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, sample_size):
        sample = []
        total_len = 0
        while True:
            sentences = list(self.sentences)
            for i, sent in enumerate(sentences):  # add new sentence until sample_size is reached
                tokenized = self.tokenizer.encode(' ' + sent, add_special_tokens=False)
                if not self.length_is_ok(tokenized):
                    continue
                total_len += len(tokenized)
                sample.append(tokenized)
                if total_len >= sample_size:
                    self.sentences = self.sentences[i+1:]
                    cutoff = total_len - sample_size
                    if cutoff > 0:
                        sample[-1] = sample[-1][:-cutoff]
                    return sample

            self.sentences = []
            self.sample_sentences_(sample_size)  # appends new sentences, can be updated to just return new sentences

    def sample_sentences_(self, sample_size):
        sentences = []
        while len(sentences) == 0:
            text = self.next_sample_()
            if self.shuffle:
                if len(text) == 0:
                    continue
                text = text[self.gen.choice(len(text)):]  # start from random position in text
                text = text[:sample_size * 10]            # cut too long texts to speed up tokenization
            sentences += self.sentence_tokenizer.tokenize(text)
            if self.shuffle:
                sentences = sentences[1:-1]
        self.sentences += sentences

    def next_sample_(self):
        if self.shuffle:
            self.total_tokens = 0
            sample_ind = self.gen.choice(len(self.dataset))
            sample = self.dataset[int(sample_ind)]['text']
        else:
            sample = self.dataset[int(self.sample_ind)]['text']
            self.sample_ind += 1
            self.sample_ind = self.sample_ind % len(self.dataset)
        return sample

    def length_is_ok(self, tokenized):
        if self.max_sentence_len is not None and len(tokenized) > self.max_sentence_len:
            return False
        if self.min_sentence_len is not None and len(tokenized) < self.min_sentence_len:
            return False
        return True


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return


def logistic(x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(
        self,
        needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir="./haystack_for_detect",
        retrieval_question="What is the best thing to do in San Francisco?",
        results_version = 1,
        context_lengths = None,
        document_depth_percent_min = 0,
        document_depth_percent_max = 100,
        document_depth_percent_intervals = 10,
        document_depth_percents = None,
        document_depth_percent_interval_type = "linear",
        model_provider = "OpenAI",
        model_name='',
        model_name_suffix=None,
        num_concurrent_requests = 1,
        save_results = True,
        save_contexts = True,
        final_context_length_buffer = 200,
        seconds_to_sleep_between_completions = None,
        print_ongoing_status = True,
        needle_ids=None,
        mask_topk=0,
        head_file="",
        tag = None,
        custom_block_list = False
    ):
        
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        needles_and_stacks = [json.loads(l) for l in open(f"{haystack_dir}/reasoning_needle.jsonl")]
        # needles_and_stacks = [json.loads(l) for l in open("/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle.jsonl")]
        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.needle_ids = needle_ids
        self.golden_answer = [l["golden_answer"] for l in needles_and_stacks]
        haystack = datasets.load_dataset("/mnt/petrelfs/tangzecheng/local_data/pg19-test", split="test")  # zecheng_note : 从pg 19预训练数据集里面加载数据作为上下文 /data/data/zecheng/data/pg19-test  ||| /mnt/petrelfs/tangzecheng/local_data/pg19-test
        self.noise_sampler_test = SentenceSampler(haystack, tokenizer=self.enc, shuffle=False, random_seed=None)
        self.needle_list = [l["needle"] for l in needles_and_stacks]
        self.retrieval_question_list = [l["question"] for l in needles_and_stacks]
        self.real_ansers_list = [l["real_needle"] for l in needles_and_stacks]
        self.tags = [l["tag"] for l in needles_and_stacks]
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.tag = tag
        # zecheng_note: conduct attention mask
        self.mask_topk = mask_topk
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.testing_results = []
        self.head_file = head_file
        self.head_counter = defaultdict(list)
        self.fail_head_counter = defaultdict(list)
        self.detected_tokens = list()
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        print("loading from %s" % model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads

        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        if "qwen" in self.model_version.lower():  # balanced_low_0
            self.model_to_test = Qwen2ForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map = "auto",use_flash_attention_2="flash_attention_2"
                ).eval()
        elif "Mixtral" in self.model_version:
            self.model_to_test = MixtralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map = "auto",use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        elif "Mistral" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map = "auto",use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        elif "Phi3" in self.model_version:
            self.model_to_test = Phi3ForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto", device_map = "auto",use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        else:
            self.model_to_test = LlamaForCausalLM.from_pretrained(
                model_name, use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16, device_map = "auto").eval()
            
        if 'llama-2-7b-80k' in self.model_version:
            scaling_factor = 10
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
            
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1
        else:
            self.multi_gpus = True
            
        self.model_to_test_description = model_name
        self.evaluation_model = None
      
        model_name = model_name.split('/')[-1]

        if self.construct_random_head:
            self.block_list = self.custom_block_list(self.layer_num, self.head_num)
        else:
            if self.mask_topk!=0:
                with open(self.head_file, "r") as file:
                    stable_block_list = json.loads(file.readline())
                stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
                stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
                self.block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:100]
                self.tags = [l + f'_{self.mask_topk}' for l in self.tags]
                if self.mask_topk > 0:
                    print(f"masking out top {self.mask_topk} retrieval heads")
                else:
                    print(f"masking out random {-self.mask_topk}  heads")
            else:
                self.block_list = []
        
        
    def custom_block_list(self, num_layers, num_heads):
        block_list = []
        for i in range(7, 15):
            random_heads = random.sample(range(num_heads), num_heads // 8)  # 从当前layer里面随机mask一半的heads
            for j in random_heads:
                block_list.append([i, j])
        return block_list

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths[::-1]:  # zecheng_note: 首先进行长的测试，然后进行短的测试 
            if context_length < args.s_len or context_length > args.e_len: 
                continue
            all_combinations = list(itertools.combinations(list(range(0, self.document_depth_percent_intervals)), len(self.real_needle)))
            for depth_percent in all_combinations:  # zecheng_note: 这里是fact插入的位置，不同的排列组合
                depth_percent = np.array(depth_percent) / self.document_depth_percent_intervals
                self.evaluate_and_log(context_length, depth_percent)

    def construct_random_head(self, n):
        results = []
        seed_list = [i  for i in range(32)]
        random.shuffle(seed_list)
        while len(results) < n:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.block_list:
                continue
            else:
                results.append((l, h))
        return results

    def retrieval_calculate(self, attention_maxtrix,retrieval_score, inp, step_token, topk=1):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                for v, i in zip(values, idx):
                    if self.check_if_attend_ref(i, retrieval_score, layer_idx, head_idx):
                        break

    def check_if_attend_ref(self, i, retrieval_score, layer_idx, head_idx):
        total_length = 0
        for pos in self.needle_pos:
            total_length += pos[1] - pos[0]

        for pos in self.needle_pos:
            if pos[0] <= i < pos[1]:  # if hit into the scope, count it
                retrieval_score[layer_idx][head_idx][0] += 1/total_length
                retrieval_score[layer_idx][head_idx][1].append(i.cpu().item())
                return True
        return False


    def retrieval_head_accumulate(self, retrieval_score, fail=False):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                if fail:
                    self.fail_head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])
                else:
                    self.head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])


    def decode(self, q_outputs, inp, decode_len, block_list=None):
        output, retrieval_score = [], [[[0, []] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        past_kv = q_outputs.past_key_values
        for step_i in trange(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch" )
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.decode(inp.item())
            output.append(inp.item())
            self.retrieval_calculate(outputs.attentions, retrieval_score, inp, step_token, topk=3)
            if step_token=='<0x0A>' or inp.item()==144: break
        return output, retrieval_score


    def find_niah_needle_idx(self, needle):
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        print( self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return [(i, i + span_len)]
        return [(-1, -1)]
    

    def find_multi_needle_idx(self, needle, with_noise=True):
        all_evi_pos = []
        for i, evi in enumerate(needle):
            needle_ids = self.enc(evi, add_special_tokens=False)["input_ids"]
            print(f"evidence {i} --> {self.enc.decode(needle_ids, skip_special_tokens=False)}")
            span_len = len(needle_ids)
            for j in range(len(self.prompt_ids)):
                token_span = self.prompt_ids[j : j + span_len]
                span_ids = set(token_span.tolist())
                overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
                if(overlap > 0.8):
                    all_evi_pos.append((j + 1, j + span_len))
                    print(f"find evidence {i} at --> {(j + 1, j + span_len)} --> {self.enc.decode(self.prompt_ids[j + 1: j + span_len], skip_special_tokens=False)}")
                    break
        return all_evi_pos


    def find_needle_idx(self, needle) -> List[Tuple[int, int]]:
        if isinstance(needle, str):
            return self.find_niah_needle_idx(needle)
        return self.find_multi_needle_idx(needle)
        

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        # Go generate the required length context and place your needle statement in
        if self.mask_topk > 0:
            block_list = self.block_list[:self.mask_topk]
            save_name = f"{self.model_version}_block_top{self.mask_topk}"
        elif self.mask_topk == 0:
            block_list = None
            save_name = self.model_version
        else:
            block_list = self.construct_random_head(-self.mask_topk)
            save_name = f"{self.model_version}_block_random{-self.mask_topk}"

        context = self.generate_context(context_length, depth_percent)
        input_context = context + f"\nQuestion: {self.retrieval_question}\nAnswer:"
        input_ids = self.enc.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, return_tensors='pt')

        # Prepare your message to send to the model you're going to evaluate
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        input_ids = input_ids.to(self.model_to_test.device)
        # self.needle_start, self.needle_end = self.find_needle_idx(self.real_needle)
        self.needle_pos = self.find_needle_idx(self.real_needle)
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            output, retrieval_score = self.decode(q_outputs, input_ids[:,-1], 50, block_list=block_list)
            response = self.enc.decode(output[:-1], skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
        # zecheng_note: 如果是niah testing，用rouge指标；如果是implicit reasoning, 用是否包含golden answer来判断
        score = 100 if self.golden_answer in response else 0

        # zecheng_note: 对于niah testing，如果分数大于50，则算retrieval head；如果是implicit reasoning且分数等于100分，则算Reasoning Head
        if score == 100:  
            self.retrieval_head_accumulate(retrieval_score)
            head_score = [(i[0], np.mean(i[1])) for i in self.head_counter.items()]
            head_score = sorted(head_score, key=lambda x:x[1], reverse=True)
            print([[i[0]] for i in head_score][:20])
        else:
            self.retrieval_head_accumulate(retrieval_score, fail=True)
            head_score = [(i[0], np.mean(i[1])) for i in self.fail_head_counter.items()]
            head_score = sorted(head_score, key=lambda x:x[1], reverse=True)
            print([[i[0]] for i in head_score][:20])
        
        all_detected_tokens = list()
        for item in retrieval_score:
            for head in item:
                all_detected_tokens.extend(head[1])
        all_detected_token_id = sorted(list(set(all_detected_tokens)))
        detected_evidences = self.enc.decode([input_ids[0][i] for i in all_detected_token_id])
            
        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : depth_percent.tolist(),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'golden_text': self.real_needle,
            'score' : score,
            'detected_token_id' : all_detected_token_id,
            'detected_evidences' : detected_evidences,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print(f"detected_evidences {detected_evidences}")
            print (f"Response: {response}\n")
        if isinstance(depth_percent, float) or isinstance(depth_percent, int) or isinstance(depth_percent, np.int64):
            context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
        else:
            combination = depth_percent * self.document_depth_percent_intervals
            combination = [int(i) for i in combination]
            tmp = "_".join(list(map(str, combination)))
            context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_combination_{tmp}'
            
        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.tag}/{self.model_version}'):
                os.makedirs(f'contexts/{self.tag}/{self.model_version}')

            with open(f'contexts/{self.tag}/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'results/{self.tag}/{self.model_version}'):
                os.makedirs(f'results/{self.tag}/{self.model_version}')
            
            # Save the result to file for retesting
            p = f'results/{self.tag}/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)


    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """
        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False


    def generate_context(self, context_length, depth_percent):
        """ zecheng_note: 如果是implicit reasoning, 则使用noise_sampler_test来采样上下文，不用原来的context文件
        """
        context = self.insert_multi_needle(depth_percent, context_length)
        return context
    

    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    

    def insert_multi_needle(self, depth_percent, context_length):
        # self.noise_sampler_test
        # zecheng note: 目前仅支持随机插入evidences, 不规定具体位置
        all_tok_needles = []
        real_tok_needles = []
        for needle in self.needle:
            all_tok_needles.append(self.encode_text_to_tokens(needle))
        for needle in self.real_needle:
            real_tok_needles.append(self.encode_text_to_tokens(needle))

        disturb_tok_needles = [i for i in all_tok_needles if i not in real_tok_needles]
        
        all_needle_length = sum([len(needle) for needle in all_tok_needles])
        tok_question = self.encode_text_to_tokens(self.retrieval_question)
        background_text_len = context_length - all_needle_length - len(tok_question)
        # context_length -= self.final_context_length_buffer
        
        background_text = self.noise_sampler_test.get_sample(background_text_len)
        disturb_pos=np.random.choice(len(background_text)+1, len(disturb_tok_needles))

        updated_sample = [[] for _ in range(len(background_text) + 1)]
        
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(real_tok_needles, real_pos):  # insert real needle
            updated_sample[pos].append(fact)
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)
        for i, s in enumerate(background_text):  # insert irrevelent needle
            updated_sample[i].append(s)

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        new_context = self.decode_tokens(tokens)

        return new_context


    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            
            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context


    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")


    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while len(context.split()) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context


    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        

    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")


    def get_results(self):
        return self.testing_results


    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle}")
        print ("\n\n")

    def start_test(self, args):
        for ni in self.needle_ids:
            self.tag = self.tags[ni]
            self.needle = self.needle_list[ni]
            self.haystack_dir = None
            self.real_needle  = self.real_ansers_list[ni]
            self.retrieval_question = self.retrieval_question_list[ni]
            self.golden_answer = self.golden_answer[ni]
            if self.print_ongoing_status:
                self.print_start_test_summary()
            self.run_test(args)
        
        if not os.path.exists(f"head_score/{self.tag}"):
            os.makedirs(f"head_score/{self.tag}")

        # if os.path.exists(f"head_score/{self.tag}/{self.model_version}.json"):
        #     with open(f"./head_score/{self.tag}/{self.model_version}.json", "r") as file:
        #         head_counter = json.loads(file.readline())
        #     for k,v in head_counter.items():
        #         self.head_counter[k] += v
        with open(f"head_score/{self.tag}/success_{self.model_version}.json", 'w') as f:
            json.dump(self.head_counter, f)
        with open(f"head_score/{self.tag}/fail_{self.model_version}.json", 'w') as f:
            json.dump(self.fail_head_counter, f)

if __name__ == "__main__":
    """
    python -m debugpy --listen 5678 --wait-for-client reasoning_head_detection.py
    """

    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--needle_ids', metavar='N', type=int, nargs='+', help='a list of numbers')
    parser.add_argument('--mask_topk', metavar='N', type=int, default=0, help='masking top K heads')
    parser.add_argument('--head_file', type=str, default=None, help='path to head file')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    args = parser.parse_args()
    
    # zecheng note: 修改完的代码必须事先输入context lengths 区间，是
    # args.model_path = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    args.model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    args.e_len = 64000
    args.s_len = 4000
    args.mask_topk = 10
    args.custom_block_list = True
    args.needle_ids = [4]
    args.head_file = "/mnt/petrelfs/tangzecheng/MyRLHF/reetrievalheaddetect/head_score/5-hop/success_Meta-Llama-3-8B-Instruct.json"
    model_name = args.model_path
    # context_lengths = np.array([4000, 8000, 16000, 32000, 64000])
    context_lengths = np.round(np.linspace(args.s_len, args.e_len, 15, endpoint=True)).astype(int)

    ht = LLMNeedleHaystackTester(
        model_name=model_name, 
        # haystack_dir="/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect",
        haystack_dir="/mnt/petrelfs/tangzecheng/MyRLHF/reetrievalheaddetect/haystack_for_detect",
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=False,
        save_results=True,
        context_lengths = context_lengths,
        final_context_length_buffer=200,
        document_depth_percents=np.array([0, 20, 40, 60, 80]),
        needle_ids=args.needle_ids,
        mask_topk=args.mask_topk,
        head_file=args.head_file,
        custom_block_list=args.custom_block_list,
        # tag="q3_inf_diff_pos"
    )

    ht.start_test(args)