from loguru import logger
import evaluate
import re
import Levenshtein
from itertools import chain
import numpy as np
from modelzipper.tutils import *

class API_Evaluator:

    def __init__(self, content, task_type) -> None:
        self.task_type = task_type
        self.content = content
        
    def extract_api_ids(self, text):
        matches = re.findall(r"<API_(\d+)>", text)
        return list(set([id  for id in matches]))

    def eval_api_res(self):
        logger.info(f"load evaluators, rouge and bleu scores")
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        
        logger.info(f"load and postprocess generated results")
        predictions, pred_ids, labels, label_ids = [], [], [], []         
        for item in self.content:
            pred_str = item['pred'][0]
            match_str = re.findall(r"<API_\d+>.*?</API_\d+>", pred_str, re.DOTALL)
            if match_str:
                predictions.append("\n".join(match_str).strip())
            else:
                predictions.append("")
            golden_match = re.search(r'<TOOL_DOC>(.*?)</TOOL_DOC>', item['source_docs'], re.DOTALL)
            golden_text = golden_match.group(1).strip()
            labels.append(golden_text)
            # extract api ids
            pred_ids_str, golden_ids_str = self.extract_api_ids(pred_str), self.extract_api_ids(golden_text)
            if len(pred_ids_str) > len(golden_ids_str):
                pred_ids_str = golden_ids_str[:len(pred_ids_str)]
            elif len(pred_ids_str) < len(golden_ids_str):
                pred_ids_str.extend([1000000] * (len(golden_ids_str) - len(pred_ids_str)))
            pred_ids.append(pred_ids_str)
            label_ids.append(golden_ids_str)
        
        logger.info(f"begin to evaluate the model predictions")
        flatten_pred_ids, flatten_golden_ids = list(chain(*pred_ids)), list(chain(*label_ids))
        bleu_score = bleu_metric.compute(predictions=flatten_pred_ids, references=flatten_golden_ids, max_order=1)  # just calculate 1 grams
        rouge_score = rouge_metric.compute(predictions=predictions, references=labels)
        edit_distance = np.array([Levenshtein.distance(pred, ref) for pred, ref in zip(predictions, labels)]).mean()

        
        return {
            "bleu_score": bleu_score,
            "rouge_score": rouge_score,
            "edit_score": edit_distance
        }


def test_api():
    all_tool_subdirs = auto_read_dir("/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct", file_suffix="api")
    tool_location_subdirs = [auto_read_dir(subdir, file_suffix="location")[0] for subdir in all_tool_subdirs]
    tool_location_files = [auto_read_dir(subdir, file_suffix="jsonl")[0] for subdir in tool_location_subdirs]

    logger.info(tool_location_files)

    all_content = dict([(os.path.basename(file).split('.')[0], auto_read_data(file)) for file in tool_location_files])

    for task, content in all_content.items():
        api_evalator = API_Evaluator(content, task)
        eval_res = api_evalator.eval_api_res()
        logger.info(f"task: {task}\n{eval_res}")
        
# def main(dataset_path, dataset_name):
#     evaluator = Evaluator(dataset_name)
#     res = evaluator.calculate_metrics()


# def test():
#     dataset_path = "/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/rapid_multiple_api/tool_location/preds_rapid_multiple_api.jsonl"
#     evaluator = Evaluator(dataset_path, task_type='api')
#     res = evaluator.calculate_metrics()


if __name__ == '__main__':
    test_api()


    # parser = argparse.ArgumentParser(description="Evaluating the location capability of LCMs")
    # parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    # parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    # args = parser.parse_args()
    
    # main(args.dataset_path)