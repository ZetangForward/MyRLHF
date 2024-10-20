import math
from abc import ABC
import os

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler
from torch.nn import functional as F
from openrlhf.datasets import SFTDataset
from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from flash_attn.utils.distributed import all_gather


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = 10000000  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        rank = self.strategy.ring_attn_rank

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompts_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                    labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)
                    
                    # print(f"local rank: {rank} before model --> inputs shape: {inputs.shape}, attention_mask shape: {attention_mask.shape}\n")
                    
                    local_logits = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=prompts_id_lens, 
                        return_output=True,
                    )["logits"]
                    
                    rank = self.strategy.ring_attn_rank
                    total_seq_len = labels.numel()
                    local_seq_len = total_seq_len // self.strategy.ring_attn_size
                    # print(f"local rank: {rank} after model --> local_seq_len: {local_seq_len}, total_seq_len: {total_seq_len}, self.strategy.ring_attn_size: {self.strategy.ring_attn_size}\n")
                    local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
                    # print(f"local rank: {rank} after model --> local_slice: {local_slice}\n")
                    
                    labels.roll(shifts=-1, dims=1)  # shift the label to the left to avoid the bos token 
                    labels[:, -1] = self.loss_fn.IGNORE_INDEX  # pad the last token with -100 to avoid the loss computation

                    local_label = labels[:, local_slice]
                    if rank == self.strategy.ring_attn_size - 1:
                    # add a dummy label to the last logit
                        local_label = F.pad(local_label, (0, 1), value=self.loss_fn.IGNORE_INDEX)
                    local_mask = (local_label == self.loss_fn.IGNORE_INDEX)  # Shape: (batch_size, seq_len)

                    # convert -100 in local_label into 0  
                    local_label[local_label==self.loss_fn.IGNORE_INDEX] = 0
                    per_token_logps = torch.gather(local_logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)).squeeze(2)
                    # print(f"local rank: {rank} after model --> per_token_logps shape: {per_token_logps.shape}\n")

                    per_token_logps *= ~local_mask
                    
                    gathered_logps = all_gather(per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
                    masked_logps_flat = gathered_logps.view(-1)

                    # Calculate the cross-entropy loss, ensuring local_label is gathered and flattened accordingly
                    gpt_loss = -torch.mean(masked_logps_flat)

                    # local_loss = self.loss_fn(local_logits, local_label)
                    # print(f"local rank: {rank} --> gpt_loss is: {gpt_loss}\n")
                    
                    # Initialize gathered_losses here, after local_loss is computed
                    # gathered_losses = [torch.zeros_like(local_loss) for _ in range(self.strategy.ring_attn_group.size())]
                    # torch.distributed.all_gather(gathered_losses, local_loss, group=self.strategy.ring_attn_group)
                    # gathered_losses = [torch.zeros_like(local_loss) for _ in range(self.strategy.ring_attn_size)]
                    # print(f"local rank: {rank} after model --> local_label: {local_label.shape}\n")
                    # # print(f"after model --> local rank: {rank}, local_logits shape: {local_logits.shape}, attention_mask shape: {attention_mask.shape}")
                    
                    
                    # print(f"before gathering, rank {rank}, local_gpt_loss is: ", local_gpt_loss)
                    # print()
                    
                    # print(f"after gathering, rank {rank} | all_loss is: {gathered_losses}\n")
                    # gpt_loss = sum(gathered_losses) / len(gathered_losses)
                    # print(f"rank: {rank}, local_label shape: {local_label.shape}, locak_label max: {local_label.max()}, logits_shape: {logits.shape}\n")
                    # local_per_token_logps = torch.gather(
                    #     logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
                    # ).squeeze(2)
                    
                    # all_logits = all_gather(logits, self.strategy.ring_attn_group).reshape((1, -1, logits.size(-1)))
                    # print(f"after ring attention gathering --> size: {all_logits.shape} max logits {all_logits.max()}, min logits {all_logits.min()}\n")

                    # labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)
                    # print(f"labels shape", labels.shape)
                    # gpt_loss = self.loss_fn(all_logits, labels)

                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                    logits = self.model(
                        inputs, attention_mask=attention_mask, 
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=prompts_id_lens, 
                        return_output=True,
                    )["logits"]
                    
                    # loss function
                    labels = torch.where(
                        attention_mask.bool(),
                        inputs,
                        self.loss_fn.IGNORE_INDEX,
                    )
                    assert logits.shape[:-1] == labels.shape
                    labels = labels[:, 1:]
                    logits = logits[:, :-1, :]
                    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                    gpt_loss = (-per_token_logps).view(-1)[attention_mask]
                    
                
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )


    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompts_id_lens):
                            labels[0][index: index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompts_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
