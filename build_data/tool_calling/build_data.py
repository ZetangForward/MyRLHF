from modelzipper.tutils import *
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Tuple
import math
import random
from transformers import AutoTokenizer
from tool_parser import *
from prompt_pool import *
from tqdm import tqdm
from loguru import logger

def read_all_apis(api_list: List, api_type: str = 'single', api_id_st: int = 0):
    unique_api_pool = []
    seen = set()

    if api_type == 'multiple':
        for api in api_list:
            api_1, api_2 = api['api_1'][0], api['api_2'][0]
            for api in [api_1, api_2]:
                identifier = (api['tool_name'], api['api_name'])
                if identifier not in seen:
                    unique_api_pool.append(api)
                    seen.add(identifier)
    else:
        for api in api_list:
            identifier = (api['tool_name'], api['api_name'])
            if identifier not in seen:
                unique_api_pool.append(api)
                seen.add(identifier)
        

    # re-allocate api_id for each api
    api_name_id_mapping = dict()
    for id_, api in enumerate(unique_api_pool):
        api['api_cnt'] = id_ + api_id_st
        api_name_id_mapping[api['api_name']] = id_ + api_id_st

    query_api_mapping = dict()  # query to API id
    for sample in api_list:
        if api_type == 'multiple':
            query_api_mapping[sample['api_1'][0]['api_name']] = api_name_id_mapping[sample['api_1'][0]['api_name']]
            query_api_mapping[sample['api_2'][0]['api_name']] = api_name_id_mapping[sample['api_2'][0]['api_name']]
        else:
            query_api_mapping[sample['api_name'] ] = api_name_id_mapping[sample['api_name']]
            
    return unique_api_pool, query_api_mapping


def build_single_api_benchmark(
        raw_content: List[Dict],
        num_samples: int,
        tokenizer: AutoTokenizer,
        save_dir: str = None,
        api_pool_nums: List[int] = [100, 200, 300],
        benchmark_type: str = "tool_calling",
        save_meta_data: bool = False    
    ):
    for api_pool_num in api_pool_nums:
        # re-calculate the num_samples since some apis are treated as the demonstration
        cur_num_samples = num_samples + math.ceil(num_samples / api_pool_num)
        bucket_pool = []
        query_bucket = [raw_content[:cur_num_samples][i: i + api_pool_num + 1] for i in range(0, cur_num_samples, api_pool_num + 1)]

        # construct a global api pool
        for query_group in tqdm(query_bucket, unit="sample"):
            current_api_pool, _ = read_all_apis(query_group, api_type='single')
            other_api_pool = [query for query in raw_content if query not in query_group]
            if len(current_api_pool) >= api_pool_num + 1:
                bucket_pool.append(current_api_pool)
                continue
            remain_api_pool, _ = read_all_apis(other_api_pool, api_type='single')
            padded_apis = random.sample(remain_api_pool, api_pool_num+1-len(current_api_pool))
            bucket_pool.append(current_api_pool + padded_apis)

        logger.info(f"current setting has {len(bucket_pool)} buckets, each has {len(bucket_pool[-1])} unique APIs")

        processed_rapid_single_api = [dict(system_prompt=None, query=[], model_output=[], call_parameters=[], source_docs=[]) for _ in range(len(bucket_pool))]
        
        for b_id, query_group in enumerate(query_bucket):
            for sample in query_group:
                tool_sample = ToolSample(sample, type="single", benchmark_type=benchmark_type, total_pool=bucket_pool[b_id])
                tmp = tool_sample.create_reason_answer_mseeage()
                if tmp:
                    if processed_rapid_single_api[b_id]['system_prompt'] is None:
                        cur_demonstration = tool_sample.create_demonstration(benchmark_type,  return_str=False)
                        system_prompt = cur_demonstration['system_prompt']
                        query = cur_demonstration['query']
                        answer = cur_demonstration['answer']
                        source_docs = cur_demonstration['source_docs']
                        if benchmark_type == "tool_calling":
                            processed_rapid_single_api[b_id]['system_prompt'] = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": answer},
                            ]
                        elif benchmark_type == "tool_location":
                            processed_rapid_single_api[b_id]['system_prompt'] = [
                               {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": source_docs},
                            ]
                        continue  # skip the first one since this case is used for demonstration
                    processed_rapid_single_api[b_id]['query'].append(tmp['query'])
                    processed_rapid_single_api[b_id]['call_parameters'].append(tmp['call_parameters'])
                    processed_rapid_single_api[b_id]['source_docs'].append(tmp['source_docs'])
        
        # convert to json format
        processed_single_apis_json = dict()
        for b_id, api_pool in enumerate(processed_rapid_single_api):
            processed_single_apis_json[str(b_id)] = api_pool  # to minimize the saving disk space

        num_cases = sum(len(api['query']) for api in processed_rapid_single_api if api['query'])
        logger.info(f"Total number of query: {num_cases}")
        
        if save_dir is not None:
            auto_save_data(processed_single_apis_json, f"{save_dir}/rapid_single_api/{benchmark_type}/num_{api_pool_num}_pool.json")

        if save_meta_data:
            total_length = 0
            for bucket_id in tqdm(processed_single_apis_json.keys(), unit="sample", desc="Calculating Length"):
                system_prompt = processed_single_apis_json[bucket_id]['system_prompt']
                
                for query in processed_single_apis_json[bucket_id]['query']:
                    if isinstance(system_prompt, dict):  # one-shot
                        message = system_prompt.append({"role": "user", "content": query})
                    else:  # zero-shot
                        message = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ]
                    total_length += len(tokenizer.apply_chat_template(conversation=message, tokenize=True, add_generation_prompt=True))

            logger.info(f"Avg Length of {api_pool_num} is {total_length / num_samples}")


def build_multiple_api_benchmark(
        raw_content: List[Dict],
        num_samples: int,
        tokenizer: AutoTokenizer,
        save_dir: str = None,
        api_pool_nums: List[int] = [100, 200, 300],
        benchmark_type: str = "tool_calling",
        save_meta_data: bool = False,
    ):
    for api_pool_num in api_pool_nums:
        # re-calculate the num_samples since some apis are treated as the demonstration
        cur_api_pool_num = api_pool_num // 2  ## since each query has two queries
        cur_num_samples = num_samples + math.ceil(num_samples / api_pool_num)
        bucket_pool = []
        query_bucket = [raw_content[:cur_num_samples][i: i + cur_api_pool_num + 1] for i in range(0, cur_num_samples, cur_api_pool_num + 1)]
        
        for query_group in tqdm(query_bucket, unit="sample"):
            current_api_pool, _ = read_all_apis(query_group, api_type='multiple')
            if len(current_api_pool) >= api_pool_num + 1:
                bucket_pool.append(current_api_pool)
                continue
            other_api_pool = [query for query in raw_content if query not in query_group]
            while len(current_api_pool) < api_pool_num + 1:
                remain_api_pool, _ = read_all_apis(other_api_pool, api_type='multiple', api_id_st=len(current_api_pool))
                padded_apis = random.sample(remain_api_pool, api_pool_num+1-len(current_api_pool))
                current_api_pool += padded_apis
            current_api_pool = current_api_pool[:api_pool_num + 1]
            bucket_pool.append(current_api_pool)

        logger.info(f"current setting has {len(bucket_pool)} buckets, each has {len(bucket_pool[-1])} unique APIs")

        processed_rapid_multiple_api = [dict(system_prompt=None, query=[], model_output=[], call_parameters=[], source_docs=[]) for _ in range(len(bucket_pool))]
        
        for b_id, query_group in enumerate(query_bucket):
            for sample in query_group:
                tool_sample = ToolSample(sample, type="multiple", total_pool=bucket_pool[b_id])
                tmp = tool_sample.create_reason_answer_mseeage()
                if tmp:
                    if processed_rapid_multiple_api[b_id]['system_prompt'] is None:
                        cur_demonstration = tool_sample.create_demonstration(benchmark_type,  return_str=False)
                        system_prompt = cur_demonstration['system_prompt']
                        query = cur_demonstration['query']
                        answer = cur_demonstration['answer']
                        source_docs = cur_demonstration['source_docs']
                        if benchmark_type == "tool_calling":
                            processed_rapid_multiple_api[b_id]['system_prompt'] = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": answer},
                            ]
                        elif benchmark_type == "tool_location":
                            processed_rapid_multiple_api[b_id]['system_prompt'] = [
                               {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": source_docs},
                            ]
                        continue  # skip the first one since this case is used for demonstration
                    processed_rapid_multiple_api[b_id]['query'].append(tmp['query'])
                    processed_rapid_multiple_api[b_id]['call_parameters'].append(tmp['call_parameters'])
                    processed_rapid_multiple_api[b_id]['source_docs'].append(tmp['source_docs'])
        
        # convert to json format
        processed_multiple_apis_json = dict()
        for b_id, api_pool in enumerate(processed_rapid_multiple_api):
            processed_multiple_apis_json[str(b_id)] = api_pool  # to minimize the saving disk space

        num_cases = sum(len(api['query']) for api in processed_rapid_multiple_api if api['query'])
        logger.info(f"Total number of query: {num_cases}")
        
        if save_dir is not None:
            auto_save_data(processed_multiple_apis_json, f"{save_dir}/rapid_multiple_api/{benchmark_type}/num_{api_pool_num}_pool.json")

        if save_meta_data:
            total_length = 0
            for bucket_id in tqdm(processed_multiple_apis_json.keys(), unit="sample", desc="Calculating Length"):
                system_prompt = processed_multiple_apis_json[bucket_id]['system_prompt']
                
                for query in processed_multiple_apis_json[bucket_id]['query']:
                    if isinstance(system_prompt, dict):  # one-shot
                        message = system_prompt.append({"role": "user", "content": query})
                    else:  # zero-shot
                        message = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ]
                    total_length += len(tokenizer.apply_chat_template(conversation=message, tokenize=True, add_generation_prompt=True))

            logger.info(f"Avg Length of {api_pool_num} is {total_length / num_samples}")


def build_parallel_api_benchmark(
        raw_content: List[Dict],
        num_samples: int,
        tokenizer: AutoTokenizer,
        save_dir: str = None,
        api_pool_nums: List[int] = [100, 200, 300],
        benchmark_type: str = "tool_calling",
        save_meta_data: bool = False,
    ):

    for api_pool_num in api_pool_nums:
        # re-calculate the num_samples since some apis are treated as the demonstration
        cur_num_samples = num_samples + math.ceil(num_samples / api_pool_num)
        bucket_pool = []
        query_bucket = [raw_content[:cur_num_samples][i: i + api_pool_num + 1] for i in range(0, cur_num_samples, api_pool_num + 1)]
        
        for query_group in tqdm(query_bucket, unit="sample"):
            current_api_pool, _ = read_all_apis(query_group, api_type='parallel')
            if len(current_api_pool) >= api_pool_num + 1:
                bucket_pool.append(current_api_pool)
                continue
            other_api_pool = [query for query in raw_content if query not in query_group]
            while len(current_api_pool) < api_pool_num + 1:
                remain_api_pool, _ = read_all_apis(other_api_pool, api_type='parallel', api_id_st=len(current_api_pool))
                padded_apis = random.sample(remain_api_pool, api_pool_num+1-len(current_api_pool))
                current_api_pool += padded_apis
            current_api_pool = current_api_pool[:api_pool_num + 1]
            bucket_pool.append(current_api_pool)

        logger.info(f"current setting has {len(bucket_pool)} buckets, each has {len(bucket_pool[-1])} unique APIs")

        processed_rapid_prallel_api = [dict(system_prompt=None, query=[], model_output=[], call_parameters=[], source_docs=[]) for _ in range(len(bucket_pool))]
        
        for b_id, query_group in enumerate(query_bucket):
            for sample in query_group:
                tool_sample = ToolSample(sample, type="parallel", total_pool=bucket_pool[b_id])
                tmp = tool_sample.create_reason_answer_mseeage()
                if tmp:
                    if processed_rapid_prallel_api[b_id]['system_prompt'] is None:
                        cur_demonstration = tool_sample.create_demonstration(benchmark_type,  return_str=False)
                        system_prompt = cur_demonstration['system_prompt']
                        query = cur_demonstration['query']
                        answer = cur_demonstration['answer']
                        source_docs = cur_demonstration['source_docs']
                        if benchmark_type == "tool_calling":
                            processed_rapid_prallel_api[b_id]['system_prompt'] = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": answer},
                            ]
                        elif benchmark_type == "tool_location":
                            processed_rapid_prallel_api[b_id]['system_prompt'] = [
                               {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": source_docs},
                            ]
                        continue  # skip the first one since this case is used for demonstration
                    processed_rapid_prallel_api[b_id]['query'].append(tmp['query'])
                    processed_rapid_prallel_api[b_id]['call_parameters'].append(tmp['call_parameters'])
                    processed_rapid_prallel_api[b_id]['source_docs'].append(tmp['source_docs'])
        
        # convert to json format
        processed_prallel_apis_json = dict()
        for b_id, api_pool in enumerate(processed_rapid_prallel_api):
            processed_prallel_apis_json[str(b_id)] = api_pool  # to minimize the saving disk space

        num_cases = sum(len(api['query']) for api in processed_rapid_prallel_api if api['query'])
        logger.info(f"Total number of query: {num_cases}")
        
        if save_dir is not None:
            auto_save_data(processed_prallel_apis_json, f"{save_dir}/rapid_parallel_api/{benchmark_type}/num_{api_pool_num}_pool.json")

        if save_meta_data:
            total_length = 0
            for bucket_id in tqdm(processed_prallel_apis_json.keys(), unit="sample", desc="Calculating Length"):
                system_prompt = processed_prallel_apis_json[bucket_id]['system_prompt']
                
                for query in processed_prallel_apis_json[bucket_id]['query']:
                    if isinstance(system_prompt, dict):  # one-shot
                        message = system_prompt.append({"role": "user", "content": query})
                    else:  # zero-shot
                        message = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ]
                    total_length += len(tokenizer.apply_chat_template(conversation=message, tokenize=True, add_generation_prompt=True))

            logger.info(f"Avg Length of {api_pool_num} is {total_length / num_samples}")

if __name__ == "__main__":

    rapid_multiple_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_multiple_api.jsonl"
    rapid_parallel_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_parallel_api.jsonl"
    rapid_single_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_single_api.jsonl"
    
    rapid_multiple_api = auto_read_data(rapid_multiple_api)
    rapid_parallel_api = auto_read_data(rapid_parallel_api)
    rapid_single_api = auto_read_data(rapid_single_api)

    num_samples = 400
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    save_dir = "/mnt/petrelfs/tangzecheng/local_data/benchmark_data"
    # build_single_api_benchmark(rapid_single_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_calling')

    # build_single_api_benchmark(rapid_single_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_location')

    # build_multiple_api_benchmark(rapid_multiple_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_calling')

    # build_multiple_api_benchmark(rapid_multiple_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_location')


    build_parallel_api_benchmark(rapid_parallel_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_calling')

    build_parallel_api_benchmark(rapid_parallel_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100], benchmark_type='tool_location')

