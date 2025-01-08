import math
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from flash_attn.utils.distributed import all_gather
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from openrlhf.models import SimPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.models import GPTLMLoss


class CDTrainer(ABC):
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
        neg_loss_weight: float = 1.0,
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
        self.neg_loss_weight = neg_loss_weight
        self.args = strategy.args
        
        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

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
            wandb.define_metric("step_log/mini_batch_step")
            wandb.define_metric("step_log/*", step_metric="step_log/mini_batch_step", step_sync=True)
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
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
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

            self.model.train()
            pos_loss_mean, neg_loss_mean = 0, 0
            short_ctx_loss_mean = 0

            for data in self.train_dataloader:  # zecheng: 这里是packing_samples的数据

                prompt_id_lens, inputs, attention_masks, neg_inputs, neg_attention_masks, infos, clue_prompt_id_lens, clue_inputs, clue_attention_masks = data

                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                neg_inputs = neg_inputs.to(torch.cuda.current_device())
                neg_attention_masks = neg_attention_masks.to(torch.cuda.current_device())
                # if clue_inputs:
                clue_inputs = clue_inputs.to(torch.cuda.current_device()).squeeze(1)
                clue_attention_mask = clue_attention_masks.to(torch.cuda.current_device()).squeeze(1)

                pos_output = self.model(
                    inputs, 
                    attention_mask=attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["input_length"],
                    cd_noise_settings={"add_noise": False},
                )

                neg_output = self.model(
                    neg_inputs, 
                    attention_mask=neg_attention_masks, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["neg_input_length"],
                    cd_noise_settings={"add_noise": True},
                )

                # loss function
                pos_labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)
                neg_labels = torch.where(neg_attention_masks.bool(), neg_inputs, self.loss_fn.IGNORE_INDEX)

                if not self.pretrain_mode:
                    index = 0
                    for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                        pos_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                        index += input_length
                    index = 0
                    for input_length, source_len in zip(infos["neg_input_length"], prompt_id_lens):
                        neg_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                        index += input_length

                pos_loss = self.loss_fn(pos_output.logits, pos_labels)
                neg_loss = self.loss_fn(neg_output.logits, neg_labels)
                total_loss = pos_loss + self.neg_loss_weight * neg_loss

                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                pos_loss_mean = pos_loss_mean * 0.9 + 0.1 * pos_loss.item()
                neg_loss_mean = neg_loss_mean * 0.9 + 0.1 * neg_loss.item()

                ### ============================= ###
                # zecheng_note: 这里加上contextual 的labels，专门计算上下文的loss
                with torch.no_grad():
                    clue_output = self.model(
                        clue_inputs, 
                        attention_mask=clue_attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["clue_input_length"],
                        cd_noise_settings={"add_noise": False},
                    )

                    clue_labels = torch.where(
                        clue_attention_mask.bool(),
                        clue_inputs,
                        self.loss_fn.IGNORE_INDEX,
                    )

                    index = 0
                    for input_length, source_len in zip(infos["clue_input_length"], clue_prompt_id_lens):
                        clue_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                        index += input_length

                    short_ctx_gpt_loss = self.loss_fn(clue_output.logits, clue_labels)
                    short_ctx_loss_mean = short_ctx_loss_mean * 0.9 + 0.1 * short_ctx_gpt_loss.item()

                logs_dict = {
                    "short_ctx_loss": short_ctx_gpt_loss.item(),
                    "pos_loss_mean": pos_loss_mean.item(),
                    "neg_loss_mean": neg_loss_mean.item(),
                    "pos_loss": pos_loss.item(),
                    "neg_loss": neg_loss.item(),
                    "short_ctx_loss_mean": short_ctx_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                if self._wandb is not None and self.strategy.is_rank_0():
                    logs = {"step_log/%s" % k: v for k, v in {**logs_dict, "mini_batch_step": step}.items()}
                    self._wandb.log(logs)
                
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

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        # logs
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
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # self.strategy.save_ckpt(
            #     self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            # )
            self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            ctx_acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                        )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens, seg_poss = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps, rejected_logps, chosen_clue_logps, rejected_clue_logps, aux_loss, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens, seg_poss
                    )

                preference_loss, chosen_reward, reject_reward, chosen_ctx_reward, rejected_ctx_reward = self.loss_fn(
                    chosen_logps, rejected_logps, chosen_clue_logps, rejected_clue_logps,
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                if chosen_ctx_reward and rejected_ctx_reward:
                    ctx_acc_sum += (chosen_ctx_reward > rejected_ctx_reward).float().mean().item()
                loss_sum += preference_loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
                "ctx_acc_mean": ctx_acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=True
        )
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        if average_log_prob:
            return logprobs_sums / loss_masks.sum(-1) 
        return logprobs_sums

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens, seg_poss=None):
        output_denoise = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        output_noise = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            cd_noise_settings = {"add_noise": True},
        )
        all_logits = output["logits"]
        all_logps, clue_logps = self._packed_get_batch_logps(  # 如果seg_poss 是 None的话，clue_logps = []
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=True,
            seg_poss=seg_poss * 2,
        )
        chosen_logps = all_logps[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps[len(packed_seq_lens) // 2 :]
        if len(clue_logps) > 0:
            chosen_clue_logps, rejected_clue_logps = clue_logps[0], clue_logps[1]
        else:
            chosen_clue_logps, rejected_clue_logps = None, None
        aux_loss = output.aux_loss if "aux_loss" in output else []
        
        return (
            chosen_logps, rejected_logps, 
            chosen_clue_logps, rejected_clue_logps, 
            aux_loss, -all_logps[: len(packed_seq_lens) // 2].mean()
        )
            

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
        seg_poss: List = None,
    ) -> torch.FloatTensor:

        if self.strategy.ring_attn_group is None:
            assert logits.shape[:-1] == labels.shape
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        else:
            rank = self.strategy.ring_attn_rank
            total_seq_len = labels.numel()
            local_seq_len = total_seq_len // self.strategy.ring_attn_size
            local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
            local_label = labels[:, local_slice]
            if rank == self.strategy.ring_attn_size - 1:
                # add a dummy label to the last logit
                local_label = F.pad(local_label, (0, 1), value=0)
            local_per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
            ).squeeze(2)
            # we may not need to all_gather the entire tensor, but it's easier to implement.
            # use the flash_attn all_gather so that the all_gather has correct backward.
            per_token_logps = all_gather(local_per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_token_logps = per_token_logps[:, :-1]

        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]

        logprobs_sums = []
        logprobs_means = []
        clue_logprobs_means = []
        index = 0
        # print(f"packed_seq_lens: {packed_seq_lens}")
        # print(f"seg_poss: {seg_poss}")
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len

            if seg_poss is not None:  # 在计算完结果的loss之后，再对中间的evidence进行计算
                seg_logprobs_means = []
                for seg_pos in seg_poss[i]:
                    st, ed = seg_pos 
                    seg_logprobs_means.append(seq[st: ed].sum() / (ed - st))
                clue_logprobs_means.append(sum(seg_logprobs_means) / len(seg_logprobs_means))
        
        if average_log_prob:
            return torch.stack(logprobs_means), clue_logprobs_means
        return torch.stack(logprobs_sums), clue_logprobs_means
