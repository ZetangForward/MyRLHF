from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from faiss_attn.source.modeling_llama import LlamaForCausalLM
from faiss_attn.source.modeling_qwen2 import Qwen2ForCausalLM
from faiss_attn.source.modeling_mixtral import MixtralForCausalLM
from faiss_attn.source.modeling_mistral import MistralForCausalLM
from faiss_attn.source.modeling_phi3 import Phi3ForCausalLM
import numpy as np
import argparse
import datasets
import itertools
import nltk
import torch
import random
from rouge_score import rouge_scorer
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from loguru import logger
from modelzipper.tutils import *
from .utils import get_random_emoji


def random_combine(ref:list, att:list):
    att_list =[[] for _ in range(len(ref) + 1)]
    for p_att in att[:-1]:
        att_list[random.randint(0,len(ref)-1)].append(p_att)
    att_list[-1].append(att[-1])
    results = [k for k in att_list[0]]
    for r, patt in zip(ref,att_list[1:]):
        results.append(r)
        results.extend(patt)
    return results


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



class LLMNeedleHaystackTester:
    def __init__(
        self,
        needle_path = None,
        context_lengths = None,
        model_name='',
        print_ongoing_status = True,
        selected_idx = None,
        combinations_number=1,
        inject_emoji_num=0,
    ):
        """
         number of combinations, recommended to be larger since some positions may be failed to create
        """
        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        haystack = datasets.load_dataset("/mnt/petrelfs/tangzecheng/local_data/pg19-test", split="test")
        self.noise_sampler_test = SentenceSampler(haystack, tokenizer=self.enc, shuffle=False, random_seed=None)
        self.print_ongoing_status = print_ongoing_status
        self.succ_head_counter = defaultdict(lambda: defaultdict(list))
        self.fail_head_counter = defaultdict(lambda: defaultdict(list))
        self.inject_emoji_num = inject_emoji_num
        self.selected_idx = selected_idx

        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: 
            self.model_version = model_name
        self.context_lengths = context_lengths
        self.combinations_number = combinations_number
        self.needle_path = needle_path

        logger.info("loading from %s" % model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads

        logger.info(f"layer number: {self.layer_num}, head number {self.head_num}")
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

    def retrieval_calculate(self, attention_maxtrix, retrieval_score, flatten_search_pos, flatten_attack_pos, irrevelant_pos, step_token_id, topk=1):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                self.check_if_attend_ref(idx, step_token_id, retrieval_score, layer_idx, head_idx, flatten_search_pos, flatten_attack_pos, irrevelant_pos)

    def check_if_attend_ref(self, attention_index: List, step_token_id: str, retrieval_score: Dict, layer_idx, head_idx, flatten_search_pos: List[int], flatten_attack_pos: List[int], irrevelant_pos: List[int]):
        """
        check if the attention index (where the model put attention at) 
        fall into the search_pos or attack_pos scope
        if yes, record those positions; otherwise, do nothing

        record rules:
        1. attention_score = number of attended times for each head
        2. attention_token_id = tokenized index of the attended token
        """
        for idx in attention_index:
            if idx in flatten_search_pos:
                retrieval_score[layer_idx][head_idx]['clue_pos']['attention_score'] += 1 / (self.layer_num * self.head_num)
                retrieval_score[layer_idx][head_idx]['clue_pos']['attention_token_id'].add(step_token_id)
            elif idx in flatten_attack_pos:
                retrieval_score[layer_idx][head_idx]['attack_pos']['attention_score'] += 1 / (self.layer_num * self.head_num)
                retrieval_score[layer_idx][head_idx]['attack_pos']['attention_token_id'].add(step_token_id)
            elif idx < irrevelant_pos[1] and idx >= irrevelant_pos[0]:
                retrieval_score[layer_idx][head_idx]['irrelevant_pos']['attention_score'] += 1 / (self.layer_num * self.head_num)
                retrieval_score[layer_idx][head_idx]['irrelevant_pos']['attention_token_id'].add(step_token_id)


    def retrieval_head_accumulate(self, retrieval_score, fail=False):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                if fail:
                    self.fail_head_counter[f"{layer_idx}-{head_idx}"]['clue_pos'].append(retrieval_score[layer_idx][head_idx]['clue_pos']['attention_score'])
                    self.fail_head_counter[f"{layer_idx}-{head_idx}"]['attack_pos'].append(retrieval_score[layer_idx][head_idx]['attack_pos']['attention_score'])
                    self.fail_head_counter[f"{layer_idx}-{head_idx}"]['irrelevant_pos'].append(retrieval_score[layer_idx][head_idx]['irrelevant_pos']['attention_score'])
                else:
                    self.succ_head_counter[f"{layer_idx}-{head_idx}"]['clue_pos'].append(retrieval_score[layer_idx][head_idx]['clue_pos']['attention_score'])
                    self.succ_head_counter[f"{layer_idx}-{head_idx}"]['attack_pos'].append(retrieval_score[layer_idx][head_idx]['attack_pos']['attention_score'])
                    self.succ_head_counter[f"{layer_idx}-{head_idx}"]['irrelevant_pos'].append(retrieval_score[layer_idx][head_idx]['irrelevant_pos']['attention_score'])

    def decode(self, q_outputs, inp, decode_len, flatten_search_pos, flatten_attack_pos, irrevelant_pos):
        output = []
        retrieval_score = [
            [
                {
                    "clue_pos": {"attention_score": 0, "attention_token_id": set()},
                    "attack_pos": {"attention_score": 0, "attention_token_id": set()},
                    "irrelevant_pos": {"attention_score": 0, "attention_token_id": set()},
                }    
            for _ in range(self.head_num)
            ] for _ in range(self.layer_num)
        ]
        past_kv = q_outputs.past_key_values
        total_steps = 0

        while total_steps < decode_len:
            total_steps += 1
            inp = inp.view(1, 1)
            outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch")
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.decode(inp.item())
            output.append(inp.item())
            self.retrieval_calculate(outputs.attentions, retrieval_score, flatten_search_pos, flatten_attack_pos, irrevelant_pos, inp.item(), topk=1)
            if step_token=='<0x0A>' or inp.item()==self.enc.eos_token_id: break

        # normalize the attention score by step numbers
        for layer_scores in retrieval_score:
            for head_scores in layer_scores:
                head_scores['clue_pos']['attention_score'] /= total_steps
                head_scores['attack_pos']['attention_score'] /= total_steps
                head_scores['irrelevant_pos']['attention_score'] /= total_steps
        return output, retrieval_score


    def find_multi_needle_idx(self, input_ids, needles):
        all_evi_pos = []
        for i, evi in enumerate(needles):
            if isinstance(evi, str):
                needle_ids = self.enc(evi, add_special_tokens=False)["input_ids"]
            else:
                needle_ids = evi
            span_len = len(needle_ids)
            for j in range(len(input_ids)):
                token_span = input_ids[j : j + span_len]
                if token_span.tolist()[1:-1] == needle_ids[1:-1]:
                # span_ids = set(token_span.tolist())
                # overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
                # if(overlap > 0.98):
                    all_evi_pos.append((j + 1, j + span_len - 1))
                    # logger.info(f"find evidence {i} at --> {(j + 1, j + span_len)} --> {self.enc.decode(input_ids[j + 1: j + span_len], skip_special_tokens=False)}")
                    break
        return all_evi_pos   

    def inject_emoji(self, inp, emoji_lst, inject_bound, avoid_pos):
        """
        emoji_lst: a list contains all injected emojis -> [[2,2,4], [91,33,11]]
        inject_bound: a list contains st and ed -> [st, ed]
        avoid_pos: a list contains all avoid postions -> [0,1,2,9,12,44]
        """
        flag = False
        if inp.dim() > 1:
            inp = inp.squeeze(0)
            flag = True
        inp = inp.tolist()
        avail_inject_pos = list(set(range(inject_bound[0], inject_bound[1])) - set(avoid_pos))
        if len(avail_inject_pos) < len(emoji_lst):
            raise ValueError("There are not enough positions to inject all emojis")
        random_injected_pos = random.sample(avail_inject_pos, len(emoji_lst))
        random_injected_pos.sort()
        
        result, emoji_injected_pos = [], []
        prev_pos = 0
        for pos, emoji in zip(random_injected_pos, emoji_lst):
            result.extend(inp[prev_pos:pos])
            result.extend(emoji)
            emoji_injected_pos.append((pos, pos + len(emoji)))
            prev_pos = pos
        result.extend(inp[prev_pos:])

        final_inp = torch.tensor(result, dtype=torch.long)
        if flag:
            final_inp = final_inp.unsqueeze(0)
        return final_inp, emoji_injected_pos

    def search_pos(self, inp, evidence, disturb):
        search_pos = self.find_multi_needle_idx(inp[0], evidence)
        attack_pos = self.find_multi_needle_idx(inp[0], disturb)
        if (len(search_pos) != len(evidence)) or (len(attack_pos) != len(disturb)):
            logger.info("length of search pos and attack pos is not equal to the length of evidence and disturb_tok_needles, skip this case")
            logger.info(f"search_pos length: {len(search_pos)} | evidence length: {len(evidence)}")
            logger.info(f"attack_pos length: {len(attack_pos)} | attack clues length: {len(disturb)}")
            return None, None, None

        irrevelant_pos = [(inp == 128007).nonzero(as_tuple=True)[1][1].item(), (inp == 128009).nonzero(as_tuple=True)[1][-1].item()]  # for llama3.1 model, context is wrapped between <|end_header_id|> (128007) and <|eot_id|> (128009) tokens
        flatten_search_pos = [num for st, ed in search_pos for num in range(st, ed + 1)]
        flatten_attack_pos = [num for st, ed in attack_pos for num in range(st, ed + 1)]

        if len(set(flatten_search_pos) & set(flatten_attack_pos)) != 0:
            logger.info("search_pos and attack_pos should not overlap")
            logger.info(f"search_pos: {search_pos}")
            logger.info(f"attack_pos: {attack_pos}")
            return None, None, None

        return flatten_search_pos, flatten_attack_pos, irrevelant_pos

    def evaluate_and_log(self, background_text, depth_percent, evidence, disturb_tok_needles, disturb_pos, question, answer, injected_emojis):
        
        if background_text is not None:
            depth_percent = [i / 10 for i in depth_percent]
            updated_sample = [[] for _ in range(len(background_text) + 1)]
            real_pos = [int(len(background_text) * i) for i in depth_percent]
            for fact, pos in zip(evidence, real_pos):  # insert real needle
                updated_sample[pos].append(fact)
            for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
                updated_sample[pos].append(fact)
            for i, s in enumerate(background_text):  # insert irrevelent needle
                updated_sample[i].append(s)
        else:
            updated_sample = random_combine(evidence[:-1], disturb_tok_needles+[evidence[-1]])
            updated_sample = [[k] for k in updated_sample]

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        new_context = self.enc.decode(tokens)
        input_context = new_context + f"\nQuestion: {question}\nAnswer:"
        inp = self.enc.apply_chat_template([{"role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
        
        flatten_search_pos, flatten_attack_pos, irrevelant_pos = self.search_pos(inp, evidence, disturb_tok_needles)
        if flatten_search_pos is None:  # search failed
            return False
        
        if len(injected_emojis) > 0:
            inp = self.inject_emoji(inp, injected_emojis, irrevelant_pos, flatten_search_pos + flatten_attack_pos)
        
        flatten_search_pos, flatten_attack_pos, irrevelant_pos = self.search_pos(inp, evidence, disturb_tok_needles)  # after injection, re-seach the positions
        if flatten_search_pos is None:  # search failed
            return False
        
        inp = inp.to(self.model_to_test.device)

        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=inp[:, :-1], use_cache=True, return_dict=True)
            output, retrieval_score = self.decode(q_outputs, inp[:, -1], 20, flatten_search_pos, flatten_attack_pos, irrevelant_pos)
            response = self.enc.decode(output[:-1], skip_special_tokens=True).strip()
        
        logger.info(f"model response: {response}")
        logger.info(f"gloden label: {answer}")
        
        score = 100 if ((answer in response) and (answer not in question)) else 0
        self.retrieval_head_accumulate(retrieval_score, fail=(score==100))

        return True


    def start_test(self):
        needles_and_stacks = auto_read_data(self.needle_path)
        needle_list = [l["needle"] for l in needles_and_stacks]
        retrieval_question_list = [l["question"] for l in needles_and_stacks]
        evidence_list = [l["real_needle"] for l in needles_and_stacks]
        golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
        tags = [l["tag"] for l in needles_and_stacks]
        if self.inject_emoji_num != 0:
            injected_emojis = get_random_emoji(self.enc, self.inject_emoji_num, True)  # list[list]
        else:
            injected_emojis = []

        if self.selected_idx is None:
            self.selected_idx = range(len(needle_list))
        
        for context_length in tqdm(self.context_lengths):
            for task_tag in tqdm(["3-hop", "4-hop"]):
                for s_id in self.selected_idx:
                    if tags[s_id] != task_tag:
                        continue
                    logger.info(f"Selected idx: {s_id}")
                    logger.info(f"Answer: {golden_answer_list[s_id]}")
                    logger.info(f"Tag: {tags[s_id]}")
                    logger.info(f"Needle: {needle_list[s_id]}")
                    logger.info(f"Real Needle: {evidence_list[s_id]}")
                    logger.info("=============================================")

                    needle = [self.enc('\n' + i + '\n', add_special_tokens=False)['input_ids'] for i in needle_list[s_id]]
                    evidence = [self.enc('\n' + i + '\n', add_special_tokens=False)['input_ids'] for i in evidence_list[s_id]]
                    question = retrieval_question_list[s_id]
                    answer = golden_answer_list[s_id]

                    if context_length > 0:
                        background_text = self.noise_sampler_test.get_sample(context_length)
                        disturb_tok_needles = [i for i in needle if i not in evidence]
                        disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))
                    else:
                        background_text = None
                        disturb_tok_needles = [i for i in needle if i not in evidence]
                        disturb_pos = None

                    all_combinations = list(itertools.combinations(list(range(10)), len(evidence)))
                    if self.combinations_number >= len(all_combinations):
                        logger.info("combinations_number is larger than or equal to the number of all combinations, auto select all combinations")
                    else:
                        all_combinations = random.sample(all_combinations, self.combinations_number)

                    logger.info(all_combinations)

                    analysis_sample_nums = 0
                    with tqdm(total=len(all_combinations)) as pbar:
                        for depth_percent in all_combinations:
                            torch.cuda.empty_cache()
                            pbar.set_description(f"Processing length: {context_length} | task: {task_tag} | depth {depth_percent}")
                            res = self.evaluate_and_log(background_text, depth_percent, evidence, disturb_tok_needles, disturb_pos, question, answer, injected_emojis)
                            if res: analysis_sample_nums += 1
                            pbar.update(1)

                # after processing all the samples in different positions, average the attention scores
                for pos_score_dict in self.succ_head_counter.values():
                    for k, pos_scores in pos_score_dict.items():
                        pos_score_dict[k] = sum(pos_scores) / len(pos_scores)
                
                for pos_score_dict in self.fail_head_counter.values():
                    for k, pos_scores in pos_score_dict.items():
                        pos_score_dict[k] = sum(pos_scores) / len(pos_scores)
                        
                # after average all the scores for #all_combinations * self.selected_idx, 
                # which means one sequence length experiment is finished, then
                # save the results and refresh the 
                #   self.succ_head_counter = defaultdict(lambda: defaultdict(list))
                #   self.fail_head_counter = defaultdict(lambda: defaultdict(list))
                merge_dict = {
                    "succ_head_counter": self.succ_head_counter,
                    "fail_head_counter": self.fail_head_counter,
                }
                auto_save_data(merge_dict, f"attention_analysis/attention_score/{task_tag}-{context_length}.json")

                # refresh the counter for the next task_tag
                self.succ_head_counter = defaultdict(lambda: defaultdict(list))
                self.fail_head_counter = defaultdict(lambda: defaultdict(list))

if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--context_lengths', type=int, nargs='+', help='A list of integers')
    parser.add_argument('--inject_emoji_num', type=int, default=0, help='A list of integers')
    args = parser.parse_args()

    ht = LLMNeedleHaystackTester(
        needle_path = "./haystack_for_detect/reasoning_needle_new.jsonl",
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        context_lengths=list(reversed(args.context_lengths)),
        # context_lengths = [1900, 3900, 7900, 11900],
        print_ongoing_status = True,
        # selected_idx=[0],  # for debug
        combinations_number=10,
        inject_emoji_num=args.inject_emoji_num,
    )

    ht.start_test()


# nohup env CUDA_VISIBLE_DEVICES=0 python reasoning_head_detection_babilong.py --context_lengths=0 > logs/0.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=0 python reasoning_head_detection_babilong.py --context_lengths=1900 > logs/1900.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=1 python reasoning_head_detection_babilong.py --context_lengths=3900 > logs/3900.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=2 python reasoning_head_detection_babilong.py --context_lengths=7900 > logs/7900.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=3 python reasoning_head_detection_babilong.py --context_lengths=11900 > logs/11900.log 2>&1 &