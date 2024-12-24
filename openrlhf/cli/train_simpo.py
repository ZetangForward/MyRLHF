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


class CustomBlendingDataset:
    def __init__(self, dataset_path, tokenizer, max_len, multiple_of=1):
        self.dataset_path = dataset_path
        self.dataset = auto_read_data(dataset_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multiple_of = multiple_of

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, item_list):
        wrap_batch = {}
        batch_size = len(item_list)
        keys = list(item_list[0].keys())
        if isinstance(item_list[0][keys[0]], list):
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(torch.tensor(item_list[i][key])) for i in range(batch_size)])
        else:
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(item_list[i][key]) for i in range(batch_size)])
        return wrap_batch
    
    def packing_collate_fn(self, item_list):
        extras = []
        chosen_ids = []
        chosen_att_masks = []
        chosen_pos_ids = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_pos_ids = []
        rejected_seq_lens = []
        index = 1
        for item in item_list:
            chosen_id, reject_id = torch.tensor(item["chosen_input_ids"]), torch.tensor(item["reject_1_input_ids"])
            chosen_pos, rejected_pos = torch.tensor(item["chosen_position_ids"]), torch.tensor(item["reject_1_position_ids"])
            chosen_mask, rejected_mask = torch.tensor(item["chosen_attention_mask"]), torch.tensor(item["reject_1_attention_mask"])

            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(chosen_mask)
            chosen_pos_ids.append(chosen_pos.flatten())
            chosen_seq_lens.append(len(chosen_id.flatten()))

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(rejected_mask)
            rejected_pos_ids.append(rejected_pos.flatten())
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1


        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_position_ids = torch.cat(chosen_pos_ids + rejected_pos_ids, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:  # padding
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras


class CustomRewardDataset(RewardDataset):
    def __init__(self, dataset_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_key = getattr(self.strategy.args, "margin_key", None)
        self.margin = None
        dataset = load_from_disk(dataset_path)
        self.train_data = dataset['train']
        self.test_data = dataset['test']

    def __getitem__(self, index):
        return self.train_data[index]
    
    def __len__(self):
        return len(self.train_data)
    
    def collate_fn(self, item_list):
        return item_list
    
    def packing_collate_fn(self, item_list):
        extras = []
        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    # zecheng_note：这里加载Lora训练，原始的模型被freeze住
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    get_tokenizer(args.pretrain, None, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    dataset = load_from_disk(args.dataset)
    train_data, eval_data = dataset['train'], dataset['validation']
    
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_dpo=True,
        multiple_of=args.ring_attn_size,
    )
    eval_dataset = RewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_dpo=True,
        multiple_of=args.ring_attn_size,
    )
    import ipdb; ipdb.set_trace()

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # strategy prepare
    model, optim, scheduler = strategy.prepare(model, optim, scheduler)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = SimPOTrainer(
        model=model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    train(args)
