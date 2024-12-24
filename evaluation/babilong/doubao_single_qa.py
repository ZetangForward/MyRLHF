


import os,time
from typing import Optional, List
from openai import OpenAI

import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import os

import pandas as pd
import numpy as np

from source.babilong.metrics import compare_answers, TASK_LABELS
# from inference.utils.babilong.single_prompts import DEFAULT_PROMPTS
from inference.utils.babilong.prompts import  DEFAULT_TEMPLATE, DEFAULT_PROMPTS,get_formatted_input
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from modelzipper.tutils import *
data_dir = "/data/zecheng/Ms-PoE/babilong/"
results_folder = './babilong_evals'
saved_model_name = 'doubao128k_single_qa'
def list_to_df(lst: List[dict],sort_index=None):
    if len(lst)==0:
        print("`list_to_df`: No datas!")
        return None
    if sort_index is not None:
        lst.sort(key=lambda x: x[sort_index])
    df={k:[] for k in lst[0].keys()}
    for ele in lst:
        for k,v in ele.items():
            df[k]+=[v]
    return pd.DataFrame(df)

# nohup python doubao_single_qa.py > doubao_single_t.log
TASKS=[
    # 'qa2',
    # 'qa3','qa4', 'qa5',
        'qa6',
        #  'qa7'
       ]
SPLIT_NAMES=[
    # '0k',
             '1k','2k','4k','8k', '16k', '32k', '64k'
             ]

use_chat_template = True
use_instruction = True
use_examples = True
use_post_prompt = True


client = init_doubao_api()

all_results = []

for task in TASKS:
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
        'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
        'template': DEFAULT_TEMPLATE,
        'chat_template': use_chat_template,
    }
    for split_name in SPLIT_NAMES:

        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)
        if task =="qa6":
            task_data= load_dataset(data_dir, split_name)['qa6']
        else:
            data = json.load(open("/data/zecheng/data/single_qa/datas.json","r"))
            task_data = data[task][split_name]['datas']
        


        for index,sample in tqdm(enumerate(task_data)):
            # target = sample['golden']
            target = sample['target']
            context = sample['input']
            question = sample['question']

            # format input text
            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                            prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                            template=prompt_cfg['template']).replace("Question",
                                                                                     "Statement")
            
            result = {
                "target"      : target,
                "question"    : question,
                "task"        : task,
                "split_name"  : split_name,
                "index"       : index
            }

            result['output'] = call_with_messages(client, model_name="doubao-pro-128k", user_query=input_text)['response']

            all_results.append(result)

results_dict=defaultdict(lambda : defaultdict(list))
for result in all_results:
    results_dict[result['task']][result['split_name']].append(result)

for task in TASKS:
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
        'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
        'template': DEFAULT_TEMPLATE,
        'chat_template': use_chat_template,
    }
    prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)


    for split_name in tqdm(SPLIT_NAMES, desc='lengths'):

        outfile = Path(f'{results_folder}/{saved_model_name}/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'{results_folder}/{saved_model_name}/{task}_{split_name}_{prompt_name}.json'

        df=list_to_df(results_dict[task][split_name],sort_index='index')
        if df is None:
            print("Missing datas: ",outfile)
            continue
        df=df[['target','output','question']]
        df.to_csv(outfile)

# nohup doubao_single_qa.py > doubao_single_test.log
