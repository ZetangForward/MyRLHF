from modelzipper.tutils import *
import argparse
from loguru import logger
import evaluate



class Evaluator:

    def __init__(self, dataset_path, task_type) -> None:
        self.task_type = task_type
        
        if task_type == "api":
            rouge_metric = evaluate.load("rouge")


    def eval_api_res(self, dataset_path, labels, rouge_metric):
        content = auto_read_data(dataset_path)
        predictions, labels = [], []        
        for item in content:
            pred_str = item['pred'][0]
            match_str = re.findall(r"<API_\d+>.*?</API_\d+>", pred_str, re.DOTALL)
            predictions.append()
            labels.append(item['source_docs'])


        



def main(dataset_path, dataset_name):

    evaluator = Evaluator(dataset_name)

    content = auto_read_data(dataset_path)

    if dataset_name == "api":
        eval_res = eval_api_res(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating the location capability of LCMs")
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    args = parser.parse_args()
    
    main(args.dataset_path)