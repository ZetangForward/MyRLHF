from loguru import logger
from rouge_score import rouge_scorer
import re
import Levenshtein
from itertools import chain
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from modelzipper.tutils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class API_Evaluator:

    def __init__(self, content, task_type) -> None:
        self.task_type = task_type
        self.content = content
        
    def extract_api_ids(self, text):
        matches = re.findall(r"<API_(\d+)>", text)
        return list(set([id  for id in matches]))


    def extract_tool_calling_res(self, s):
        try:
            # Extracting ANSWER section
            plan_match = re.search(r"<PLAN>(.*?)</PLAN>", s, re.DOTALL)
            answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", s, re.DOTALL)

            if answer_match is None:
                return None

            plan = plan_match.group(1).strip().split('\n')
            answer_content = answer_match.group(1)

            # Extract API blocks using a pattern that includes parameters
            api_blocks_pattern = r"(<API_\d+>.*?</API_\d+>.*?)(?=<API_\d+>|$)"
            api_blocks = re.findall(api_blocks_pattern, answer_content, re.DOTALL)

            # Define patterns for extracting API name and parameters
            api_id_name_pattern = r"<API_(\d+)>\s*(.*?)\s*</API_\d+>"
            param_value_pattern = r"<PARAM>\s*(.*?)\s*</PARAM>\s*<VALUE>\s*(.*?)\s*</VALUE>"

            result = []
            for api_block in api_blocks:
                # Extract API id and name
                api_id_name = re.search(api_id_name_pattern, api_block)
                if api_id_name:
                    api_id, api_name = api_id_name.groups()
                else:
                    continue

                # Extract parameters and values associated with this API block
                params_values = re.findall(param_value_pattern, api_block)
                call_parameters = [{param: value} for param, value in params_values]

                result.append({
                    "api_id": api_id,
                    "api_name": api_name,
                    "call_parameter": call_parameters
                })

            # Creating the final JSON structure
            return {
                "plan": plan,
                "pred_param": result
            }
        except Exception as e:
            return None


    def get_retrieval_res(self):
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

        return {
            "predictions": predictions,
            "pred_ids": pred_ids,
            "labels": labels,
            "label_ids": label_ids
        }


    def eval_api_res(self):
        rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        retrieval_res = self.get_retrieval_res()
        predictions, pred_ids, labels, label_ids = retrieval_res['predictions'], retrieval_res['pred_ids'], retrieval_res['labels'], retrieval_res['label_ids']
        logger.info(f"begin to evaluate the model predictions")
        flatten_pred_ids, flatten_golden_ids = list(chain(*pred_ids)), list(chain(*label_ids))
        precision = precision_score(flatten_golden_ids, flatten_pred_ids, average='macro', zero_division=0)
        recall = recall_score(flatten_golden_ids, flatten_pred_ids, average='macro', zero_division=0)
        f1_scores = f1_score(flatten_golden_ids, flatten_pred_ids, average='macro', zero_division=0)
        rouge_results = [rouge_metric.score(pred_id, reference_id) for pred_id, reference_id in zip(predictions, labels)]

        # Initialize accumulators
        rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0

        # Accumulate F1 scores
        for result in rouge_results:
            rouge1_f1 += result['rouge1'].fmeasure
            rouge2_f1 += result['rouge2'].fmeasure
            rougeL_f1 += result['rougeL'].fmeasure

        # Calculate the mean
        num_results = len(rouge_results)
        rouge1_f1_mean = rouge1_f1 / num_results
        rouge2_f1_mean = rouge2_f1 / num_results
        rougeL_f1_mean = rougeL_f1 / num_results

        rouge_scores = {"ROUGE-1": rouge1_f1_mean, "ROUGE-2": rouge2_f1_mean, "ROUGE-3": rougeL_f1_mean}

        # edit_distance = np.array([Levenshtein.distance(pred, ref) for pred, ref in zip(predictions, labels)]).mean()

        return {"precision": precision, "recall": recall, "f1_score": f1_scores, "rouge_score": rouge_scores}
    

    def calculate_f1(self, true, pred):
        true_set, pred_set = set(true), set(pred)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        return precision, recall, f1

    def evaluate_model_output(self):
        api_name_recall = 0
        api_id_correct = 0
        param_name_precision = 0
        param_name_recall = 0
        param_name_f1 = 0
        value_precision = 0
        value_recall = 0
        value_f1 = 0
        

        for item in self.content:
            query, call_parameters, pred = item['query'], item['call_parameters'], item['prediction']
            

            


        for model, golden in zip(model_results, golden_results):
            # Check API Name Recall
            if model['api_name'] in [g['api_name'] for g in golden_results]:
                api_name_recall += 1

            # Check API ID Accuracy
            api_id_correct += int(int(model['api_id']) == golden['api_id'])

            # Check call_parameter names
            model_param_names = [list(param.keys())[0] for param in model['call_parameter']]
            golden_param_names = list(golden['call_parameter'].keys())
            
            p_precision, p_recall, p_f1 = self.calculate_f1(golden_param_names, model_param_names)
            param_name_precision += p_precision
            param_name_recall += p_recall
            param_name_f1 += p_f1

            # Check call_parameter values
            model_param_values = [list(param.values())[0] for param in model['call_parameter']]
            golden_param_values = list(golden['call_parameter'].values())
            
            v_precision, v_recall, v_f1 = self.calculate_f1(golden_param_values, model_param_values)
            value_precision += v_precision
            value_recall += v_recall
            value_f1 += v_f1

        # Calculate averages
        recall_average = api_name_recall / num_samples
        api_id_accuracy = api_id_correct / num_samples
        param_name_precision_average = param_name_precision / num_samples
        param_name_recall_average = param_name_recall / num_samples
        param_name_f1_average = param_name_f1 / num_samples
        value_precision_average = value_precision / num_samples
        value_recall_average = value_recall / num_samples
        value_f1_average = value_f1 / num_samples

        return {
            "api_name_recall": recall_average,
            "api_id_accuracy": api_id_accuracy,
            "param_name_precision": param_name_precision_average,
            "param_name_recall": param_name_recall_average,
            "param_name_f1": param_name_f1_average,
            "value_precision": value_precision_average,
            "value_recall": value_recall_average,
            "value_f1": value_f1_average
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


if __name__ == '__main__':
    test_api()


    # parser = argparse.ArgumentParser(description="Evaluating the location capability of LCMs")
    # parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    # parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    # args = parser.parse_args()
    
    # main(args.dataset_path)