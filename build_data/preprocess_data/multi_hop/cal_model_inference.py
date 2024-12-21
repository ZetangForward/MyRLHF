import argparse
import json
import multiprocessing as mp
import os
import sys
import copy
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from modelzipper.tutils import *
from datasets import load_dataset
from loguru import logger
import subprocess
import itertools
import random
from multiprocessing import Pool, cpu_count

def get_gpu_memory():
    """
    获取所有GPU的显存使用情况
    返回: [(GPU ID, 已用显存, 总显存), ...]
    """
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader']
        )
        lines = output.decode().strip().split('\n')
        return [tuple(map(int, line.split(','))) for line in lines]
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_free_gpu(threshold=300):  # threshold单位为MB
    """
    获取空闲的GPU ID
    threshold: 小于此显存使用量(MB)的GPU被认为是空闲的
    """
    gpu_memory = get_gpu_memory()
    if not gpu_memory:
        print("No GPUs available.")
        return []
    empty_gpus = []
    for gpu_id, memory_used, memory_total in gpu_memory:
        if memory_used < threshold:
            empty_gpus.append(gpu_id)
            print(f"GPU {gpu_id} is available: {memory_used}MB/{memory_total}MB used")
        else:
            print(f"GPU {gpu_id} is busy: {memory_used}MB/{memory_total}MB used")
    return empty_gpus



def worker(gpu_ids: str, prompts_chunk, model_path, model_args, inference_args, return_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    llm = LLM(model=model_path, **model_args)
    sampling_params = SamplingParams(**inference_args)
    chunk_message = [item['message'] for item in prompts_chunk]

    logger.info(f"Start to generate {len(chunk_message)} samples")
    outputs = llm.generate(chunk_message, sampling_params=sampling_params, use_tqdm=True)

    results = []
    for i, output in enumerate(outputs):
        generations = [o.text for o in output.outputs]
        original_prompt_data = prompts_chunk[i]  # 获取原始数据
        original_prompt_data['pred'] = generations  # 插入生成的响应
        original_prompt_data.pop('message')
        results.append(original_prompt_data)

    return_list.extend(results)


def process_item(item, drop_num=1):
    # zecheng_note: 随机丢掉drop_num个evidence
    selected_ids = random.sample(range(len(item['clue_docs'])), len(item['clue_docs']) - drop_num)
    selected_clues_docs = [item['clue_docs'][i] for i in selected_ids]
    selected_clue_pos = [item['clue_pos'][i] for i in selected_ids]
    concat_content = item['concat_content']

    # zecheng_note: 首先截断长度过长的结果
    for p, doc in zip(selected_clue_pos, selected_clues_docs):
        concat_content[p].append(doc['content'])
    concat_content = list(itertools.chain(*concat_content))
    concat_content_str = '\n'.join(concat_content)

    question, answer = item['question'], item['answer']
    instruction_format = item['instruction_format']
    prompt = instruction_format.format(concat_content=concat_content_str, q=question)

    return prompt, answer

def process_data_item(args):
    item, tokenizer, max_length = args
    prompt, answer = process_item(item)
    input_data = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
        truncation=True, max_length=max_length
    )

    input_data = tokenizer.decode(input_data)

    meta_data = copy.deepcopy(item)
    meta_data.pop('concat_content')
    return {
        "message": input_data,
        "answer": answer,
        "meta_data": meta_data
    }

def process_data(content, tokenizer, max_length=63500, num_workers=24):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Default: use all CPUs except one

    # Prepare arguments for process_data_item
    args = [(item, tokenizer, max_length) for item in content]

    # Use multiprocessing to parallelize processing
    with Pool(num_workers) as pool:
        all_inference_content = list(tqdm(pool.imap(process_data_item, args), total=len(content)))

    return all_inference_content


class Args:
    def __init__(self, platform):
        self.model_args = {
            "tensor_parallel_size": 1, 
            "gpu_memory_utilization": 0.98,
            "swap_space": 12,
            "max_model_len": 96000, 
            "trust_remote_code": True, 
        }
        self.inference_args = dict(
            n = 1, 
            temperature = 0.7, 
            max_tokens = 512, 
            seed = 42, 
            top_p = 0.95,
        )
        self.num_gpus = 8
        self.tp_size = 1

        if platform == 'pjlab':
            self.pjlab()
        else:
            self.h20()

    def h20(self):
        self.model_path = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
        self.dataset_path = "/data/zecheng/data/processed_multi_hop/filter_en"
        self.out_file_path = "/data/zecheng/data/processed_multi_hop/random_drop_1"

    def pjlab(self):
        self.model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.dataset_path = "/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/filter_en"
        self.out_file_path = "/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop_1"



if __name__ == "__main__":
    args = Args("pjlab")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    all_file_names = auto_read_dir(args.dataset_path)
    content = []
    for file_name in all_file_names:
        content.extend(auto_read_data(os.path.join(args.dataset_path, file_name)))
    logger.info(f"length of content {len(content)}")
    input_queries = process_data(content, tokenizer, num_workers=64)
    chunk_num = args.num_gpus // args.tp_size
    chunk_size = (len(input_queries) + chunk_num - 1) // chunk_num
    prompts_chunks = [input_queries[i*chunk_size:(i+1)*chunk_size] for i in range(chunk_num)]

    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    # avail_gpu_ids = get_empty_gpus()
    avail_gpu_ids = list(range(8))
    avail_gpu_ids = avail_gpu_ids[:len(avail_gpu_ids)//args.tp_size * args.tp_size]
    if len(avail_gpu_ids) == 0:
        logger.error("No available GPUs.")
        exit(1)
    
    # construct gpu_ids list
    if args.tp_size == 1:
        gpu_id_lst = [str(i) for i in range(args.num_gpus)]
    else:
        gpu_id_lst = []
        for j in range(0, len(avail_gpu_ids), args.tp_size):
            tmp = [avail_gpu_ids[i + j] for i in range(args.tp_size)]
            gpu_id_lst.append(", ".join([str(i) for i in tmp]))
    
    
    # worker(gpu_id_lst[0], prompts_chunks[0], args.model_path, args.model_args, args.inference_args, return_list)  # FIXME: Debug
    
    # 使用 tqdm 显示总进度
    for chunk_id, gpu_ids in enumerate(gpu_id_lst):
        p = mp.Process(target=worker, args=(gpu_ids, prompts_chunks[chunk_id], args.model_path, args.model_args, args.inference_args, return_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # 保存生成结果
    logger.info('Have collected ', len(return_list), 'samples, begin to save ...')
    normal_list = list(return_list)
    auto_mkdir(args.out_file_path)
    auto_save_data(normal_list, os.path.join(args.out_file_path, "infernet_res.pkl"))




    