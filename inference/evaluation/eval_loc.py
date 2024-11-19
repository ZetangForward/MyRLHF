from modelzipper.tutils import *
import argparse
from loguru import logger
import evaluate
import re
import Levenshtein
from itertools import chain


class Evaluator:

    def __init__(self, dataset_path, task_type) -> None:
        self.task_type = task_type
        self.dataset_path = dataset_path

    def calculate_metrics(self):
        if self.task_type == "api":
            eval_res = self.eval_api_res(self.dataset_path)
        else:
            pass
        return eval_res

    def extract_api_ids(self, text):
        matches = re.findall(r"<API_(\d+)>", text)
        return list(set([id  for id in matches]))

    def eval_api_res(self, dataset_path):
        logger.info(f"load evaluators, rouge and bleu scores")
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        

        logger.info(f"load and postprocess generated results")
        content = auto_read_data(dataset_path)
        predictions, pred_ids, labels, label_ids = [], [], [], []         
        for item in content:
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
        f1_scores = bleu_metric.compute(predictions=flatten_pred_ids, references=flatten_golden_ids)
        rouge_scores = rouge_metric.compute(predictions=predictions, references=labels)
        edit_distances = [Levenshtein.distance(pred, ref) for pred, ref in zip(predictions, labels)]

        logger.info(f"f1 scores" )
        import pdb; pdb.set_trace()

        
def main(dataset_path, dataset_name):
    evaluator = Evaluator(dataset_name)
    res = evaluator.calculate_metrics()


def test():
    dataset_path = "/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/rapid_multiple_api/tool_location/preds_rapid_multiple_api.jsonl"
    evaluator = Evaluator(dataset_path, task_type='api')
    res = evaluator.calculate_metrics()


if __name__ == '__main__':
    test()


    # parser = argparse.ArgumentParser(description="Evaluating the location capability of LCMs")
    # parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    # parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    # args = parser.parse_args()
    
    # main(args.dataset_path)