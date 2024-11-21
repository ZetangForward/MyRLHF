from modelzipper.tutils import *
from pprint import pprint
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Tuple
import sys
import random
import json
import re
sys.path.append('/mnt/petrelfs/tangzecheng/MyRLHF/evaluation/tool')
from eval_loc import API_Evaluator


class post_process_api_gen_res:

    def __init__(self, prediction_dir, benchmark_dir, task, prediction_path=None, save_dir=None) -> None:
        self.prediction_dir = prediction_dir
        self.benchmark_dir = benchmark_dir
        self.prediction_path = prediction_path
        self.task = task
        self.get_benchmark_data()


    def get_benchmark_data(self):
        all_tool_subdirs = auto_read_dir(self.benchmark_dir, file_suffix="api")
        tool_location_subdirs = [auto_read_dir(subdir, file_suffix="location")[0] for subdir in all_tool_subdirs]
        tool_calling_subdirs = [auto_read_dir(subdir, file_suffix="calling")[0] for subdir in all_tool_subdirs]
        self.benchmark_data_paths = dict(
            tool_location = dict([(subdir_path.split('/')[-2], subdir_path) for subdir_path in tool_location_subdirs]),
            tool_calling = dict([(subdir_path.split('/')[-2], subdir_path) for subdir_path in tool_calling_subdirs]),
        )[self.task]


    def match_pred_benchmark(self, benchmark_data_paths: Dict, model_predictions, retrieval_res):
        benchmark_data = dict([(test_setting, auto_read_data(path)) for test_setting, path in benchmark_data_paths.items()])

        for test_setting, content in benchmark_data.items():
            for bucket_id, bucket_content in content.items():
                bucket_content['retrieval_res'] = []

        for idx, item in enumerate(model_predictions):
            bucket_id, query, testing_setting, pred = item['bucket_id'], item['query'], item['testing_setting'], item['pred']
            
            # search for corresponding benchmark_data_path
            benchmark_samples = benchmark_data[testing_setting][bucket_id]
            assert benchmark_samples['query'][idx] == query, f"{benchmark_samples[idx]['query']} != {query}"
            benchmark_samples['retrieval_res'].append(retrieval_res[idx])
        return benchmark_data

    def concate_retrieval(self):
        all_tool_subdirs = auto_read_dir(self.prediction_dir, file_suffix="api")
        # all_tool_subdirs = auto_read_dir("/mnt/petrelfs/tangzecheng/local_data/inference_results/Qwen-2-5-7b-instruct", file_suffix="api")
        tool_location_subdirs = [auto_read_dir(subdir, file_suffix="location")[0] for subdir in all_tool_subdirs]
        tool_location_files = [auto_read_dir(subdir, file_suffix="jsonl")[0] for subdir in tool_location_subdirs]

        logger.info(tool_location_files)

        all_content = dict([(os.path.basename(file).split('.')[0], auto_read_data(file)) for file in tool_location_files])
        all_res = {}
        retrieval_res = {}

        for task, content in all_content.items():
            logger.info(f"task: {task} | length content: {len(content)}")

            # model predictions
            api_processor = API_Evaluator(content, task)
            retrieval_res = api_processor.get_retrieval_res()["predictions"]

            # find the corresponding benchmark data
            corresponding_benchmark_dir = self.benchmark_data_paths[task.split('preds_')[-1]]
            all_benchmark_data_paths = dict([(file_name.split('.')[0], os.path.join(corresponding_benchmark_dir, file_name)) for file_name in auto_read_dir(corresponding_benchmark_dir)])
            
            # add retrieval res into the benchmark data
            new_benchmark_data = self.match_pred_benchmark(all_benchmark_data_paths, content, retrieval_res)
            
            


if __name__ == "__main__":
    processor = post_process_api_gen_res(
        "/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct",
        "/mnt/petrelfs/tangzecheng/local_data/benchmark_data", 
        "tool_location", 
        save_dir="/mnt/petrelfs/tangzecheng/local_data/first_retrieval_res"
    )
    processor.concate_retrieval()




