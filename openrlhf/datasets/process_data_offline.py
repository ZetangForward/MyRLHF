import argparse
import math
import os
import torch
from modelzipper.tutils import *
from collections import OrderedDict
from copy import deepcopy
from datasets import load_from_disk
from datetime import datetime
from transformers.trainer import get_scheduler
from openrlhf.datasets import RewardDataset
from openrlhf.models import Actor
from openrlhf.trainer import SimPOTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
import torch.nn.functional as F



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    dataset = load_from_disk(args.dataset)
    train_data, eval_data = dataset['train'], dataset['validation']
    strategy = get_strategy(args)
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_dpo=True,
        num_processors=args.num_processors,
        multiple_of=args.ring_attn_size,
    )



class Args:
    def __init__(self):
        self.dataset = '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_data/merge_v1'
        self.pretrain = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.num_processors = 2
        self.max_len = 96000
        self.input_template = None
        self.num_processors = 2
        self.ring_attn_size = 4
        self.zero_stage = 1
        self.prompt_key = "prompt"
        self.chosen_key = "chosen"
        self.rejected_key = "rejected"


if __name__ == "__main__":
    args = Args()
    main(args)