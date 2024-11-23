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


class API_BUILDER:

    def __init__(self, raw_content: List[Dict], num_samples: int, tokenizer: AutoTokenizer, 
                save_dir: str = None, api_pool_nums: List[int] = [100, 200, 300], 
                benchmark_type: str = "tool_calling", api_type: str = 'single_api', 
                save_meta_data: bool = False) -> None:
        
        self.raw_content = raw_content
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.api_pool_nums = api_pool_nums
        self.benchmark_type = benchmark_type
        self.save_meta_data = save_meta_data
        self.api_type = api_type

    def set_benchmark_type(self, benchmark_type: str) -> None:
        self.benchmark_type = benchmark_type

    def set_api_type(self, api_type: str) -> None:
        self.api_type = api_type

    def calculate_api_bucket_nums(self, num_samples, api_pool_num):
        if self.api_type == "multiple_api":
            return num_samples + math.ceil(num_samples / (api_pool_num // 2))
        return num_samples + math.ceil(num_samples / api_pool_num)


    def build_retrieval_then_gen_data(self, save_to_disk=False):
        pass
    
    def build_api_benchmark(self, save_to_disk=False):
        for api_pool_num in self.api_pool_nums:
            # re-calculate the num_samples since some apis are treated as the demonstration
            cur_num_samples = self.calculate_api_bucket_nums(self.api_type, api_pool_num)
            bucket_pool = []
            query_bucket = [self.raw_content[:cur_num_samples][i: i + api_pool_num + 1] for i in range(0, cur_num_samples, api_pool_num + 1)]
            # construct a global api pool
            for query_group in tqdm(query_bucket, unit="sample"):
                current_api_pool, _ = read_all_apis(query_group, api_type=self.api_type)
                other_api_pool = [query for query in self.raw_content if query not in query_group]
                if len(current_api_pool) >= api_pool_num + 1:
                    bucket_pool.append(current_api_pool)
                    continue
                remain_api_pool, _ = read_all_apis(other_api_pool, api_type=self.api_type)
                padded_apis = random.sample(remain_api_pool, api_pool_num+1-len(current_api_pool))
                bucket_pool.append(current_api_pool + padded_apis)
            
            logger.info(f"current setting has {len(bucket_pool)} buckets, each has {len(bucket_pool[-1])} unique APIs")

            processed_apis = [dict(system_prompt=None, query=[], model_output=[], call_parameters=[], source_docs=[]) for _ in range(len(bucket_pool))]

            for b_id, query_group in enumerate(query_bucket):
                for sample in query_group:
                    tool_sample = ToolSample(sample, type=self.api_type, benchmark_type=self.benchmark_type, total_pool=bucket_pool[b_id])
                    tmp = tool_sample.create_reason_answer_mseeage()
                    if tmp:
                        if processed_apis[b_id]['system_prompt'] is None:
                            cur_demonstration = tool_sample.create_demonstration(self.benchmark_type,  return_str=False)
                            system_prompt = cur_demonstration['system_prompt']
                            query = cur_demonstration['query']
                            answer = cur_demonstration['answer']
                            source_docs = cur_demonstration['source_docs']
                            if self.benchmark_type == "tool_calling":
                                processed_apis[b_id]['system_prompt'] = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": query},
                                    {"role": "assistant", "content": answer},
                                ]
                            elif self.benchmark_type == "tool_location":
                                processed_apis[b_id]['system_prompt'] = [
                                {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": query},
                                    {"role": "assistant", "content": f"<TOOL_DOC>{source_docs}</TOOL_DOC>"},
                                ]
                            continue  # skip the first one since this case is used for demonstration
                        processed_apis[b_id]['query'].append(tmp['query'])
                        processed_apis[b_id]['call_parameters'].append(tmp['call_parameters'])
                        processed_apis[b_id]['source_docs'].append(tmp['source_docs'])

            # convert to json format
            processed_apis_json = dict()
            for b_id, api_pool in enumerate(processed_apis):
                processed_apis_json[str(b_id)] = api_pool  # to minimize the saving disk space

            num_cases = sum(len(api['query']) for api in processed_apis if api['query'])
            logger.info(f"Total number of query: {num_cases}")
        
        if save_to_disk:
            auto_save_data(processed_apis_json, f"{self.save_dir}/{self.api_type}/{self.benchmark_type}/num_{api_pool_num}_pool.json")
        return processed_apis_json


if __name__ == "__main__":
    rapid_multiple_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_multiple_api.jsonl"
    rapid_parallel_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_parallel_api.jsonl"
    rapid_single_api = "/mnt/hwfile/opendatalab/tangzecheng/transfer_data/llama3.1_generated_9.24/rapid_single_api.jsonl"
    
    rapid_multiple_api = auto_read_data(rapid_multiple_api)
    rapid_parallel_api = auto_read_data(rapid_parallel_api)
    rapid_single_api = auto_read_data(rapid_single_api)

    num_samples = 400
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    save_dir = "/mnt/hwfile/opendatalab/tangzecheng/benchmark_data"

    single_api_builder = API_BUILDER(rapid_single_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100])
    parallel_api_builder = API_BUILDER(rapid_parallel_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100])
    multiple_api_builder = API_BUILDER(rapid_multiple_api, num_samples, tokenizer, save_dir=save_dir, api_pool_nums=[400, 300, 200, 100])

    builders = [single_api_builder, parallel_api_builder, multiple_api_builder]
    benchmark_types = ['tool_calling', 'tool_location']
    api_types = ['single_api', 'parallel_api', 'multiple_api']

    for b_t in benchmark_types:
        for builder in builders:
            for api_type in api_types:
                builder.set_benchmark_type(b_t)
                builder.set_api_type(api_type)
                builder.build_api_benchmark(save_to_disk=True)

        
