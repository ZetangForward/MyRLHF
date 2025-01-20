import os
from abc import ABC
import torch
from typing import Dict, List, Optional, Tuple, Union
import torch.distributed as dist
from flash_attn.utils.distributed import all_gather
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm
from openrlhf.models import GPTLMLoss, SimPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


def perturb_partial_sequence(sequence, epsilon, perturb_positions):
    raise NotImplementedError("Only support pretrain mode")

    clue_poss = infos["clue_poss"]
    non_zero_mask = embeddings.grad.abs().sum(dim=2) != 0  # (b, l)
    grad_l2_norm = torch.norm(embeddings.grad, p=2, dim=2)  # (b, l)
    batch_size, seq_len, _ = embeddings.size()
    grad_mask = torch.zeros(batch_size, seq_len, device=embeddings.device)
    
    for batch_idx in range(batch_size):
        non_zero_positions = torch.where(non_zero_mask[batch_idx])[0].tolist()
        cur_clue_pos = clue_poss[batch_idx]
        clue_positions_set = set()
        for start, end in cur_clue_pos:
            clue_positions_set.update(range(start, end))
            grad_mask[batch_idx, start: end] = 1
        non_zero_positions_set = set(non_zero_positions)
        overlap = clue_positions_set & non_zero_positions_set
        overlap_count = len(overlap)
        total_clue_positions = len(clue_positions_set)
        overlap_ratio = overlap_count / total_clue_positions if total_clue_positions > 0 else 0
        # print(f"cur_clue_pos is {cur_clue_pos}")
        # print(f"Batch {batch_idx}: Non-zero positions in l -> {non_zero_positions}")
        # print(f"total_clue_positions is {total_clue_positions}")
        # print(f"overlap_count is {overlap_count}")
        # print(f"inputs.shape is {inputs.shape}")
        # print(f"non_zero_mask.shape is {non_zero_mask.shape}")
        # print(f"overlap_ratio is {overlap_ratio}")

    context_grad = embeddings.grad.clone().detach()
    clue_grad = embeddings.grad.clone().detach()

    context_grad[grad_mask.bool()] = 0  # 将 clue 部分的梯度设为 0
    context_adv_embeddings = embeddings + self.adv_epsilon * context_grad.sign()

    clue_grad = torch.where(clue_grad.sign() == 0, torch.tensor(1, device=clue_grad.device), clue_grad.sign())  # 需要加上一个极小的扰动
    clue_grad[~grad_mask.bool()] = 0  # 将非 clue 部分的梯度设为 0
    clue_adv_embeddings = embeddings + self.adv_epsilon * clue_grad.sign()


def perturb_whole_sequence(sequence, epsilon):
    """ 
    Perturb the entire sequence based on the gradient of the sequence.

    Larger gradients result in smaller perturbations, while smaller gradients
    result in larger perturbations. The noise is injected in the direction of
    the gradient (or opposite depending on gradient size relative to the average).

    Args:
        sequence (torch.Tensor): The sequence to perturb, with shape (b, l, d), 
                                 where `b` is the batch size, `l` is the sequence length, 
                                 and `d` is the embedding dimension.
        epsilon (float): The maximum possible perturbation value. Determines the 
                         scaling factor for the noise.

    Returns:
        torch.Tensor: The perturbed sequence with the same shape as the input `sequence` (b, l, d).

    Notes:
        - The gradient of the sequence should be computed prior to calling this function.
        - Noise is scaled based on the L2 norm of the gradient, with larger gradients receiving
          smaller perturbations and smaller gradients receiving larger perturbations.
        - Perturbations are applied in the direction of the gradient for smaller gradients
          and in the opposite direction for larger gradients.
    """
    
    # Calculate the L2 norm of the gradient for each position in the sequence
    grad_l2_norm = torch.norm(sequence.grad, p=2, dim=2)  # (b, l)
    sigma = epsilon / (1 + grad_l2_norm)  # (b, l)
    avg_grad_l2_norm = grad_l2_norm.mean(dim=-1, keepdim=True)  # (b, 1)
    
    # Create a mask indicating whether each position's gradient is larger than the average
    is_large_gradient = grad_l2_norm > avg_grad_l2_norm  # (b, l)
    grad_sign = sequence.grad.sign()  # (b, l, d)
    
    # Generate Gaussian noise with the same shape as the sequence
    noise = torch.randn_like(sequence, device=sequence.device)  # (b, l, d)
    noise = noise * grad_sign  # (b, l, d)
    
    # Apply perturbation in the direction based on gradient size
    # For large gradients, perturb in the opposite direction
    # For small or equal gradients, perturb in the same direction
    perturbation = torch.where(
        is_large_gradient.unsqueeze(-1),  # (b, l, 1)
        -sigma.unsqueeze(-1) * noise,    # Opposite direction for large gradients
        sigma.unsqueeze(-1) * noise      # Same direction for small gradients
    )
    
    return sequence + perturbation


class FDSMTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
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
        adv_epsilon: float = 0.1,
        sft_weight: float = 0.5,
        gan_weight: float = 1.0,
        beta: float = 0.01,
        gamma_beta_ratio: float = 0.5,
        label_smoothing: float = 0.0,
        construct_with_lora: bool = False,
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
        self.adv_epsilon = adv_epsilon
        self.sft_weight = sft_weight
        self.gan_weight = gan_weight
        self.construct_with_lora = construct_with_lora

        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)
        self.simpo_loss_fn = SimPOLoss(beta, torch.cuda.current_device(), label_smoothing, gamma_beta_ratio)
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

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

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
            loss_mean, overlap_mean, acc_mean = 0, 0, 0
            for data in self.train_dataloader:
                prompt_id_lens, inputs, attention_masks, infos, _, _, _ = data
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                ### ======= obtain the embedding gradient of the base model ====== ###
                for param in self.model.model.parameters():
                    param.requires_grad = False

                embedding_layer = self.model.model.base_model.model.get_input_embeddings()
                embeddings = embedding_layer(inputs)
                embeddings.requires_grad = True
                
                adv_output = self.model.forward_embedding(
                    inputs,
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["input_length"],
                )
                adv_loss = self.loss_fn(adv_output.logits, labels)
                adv_loss.backward()
                
                if not self.pretrain_mode:
                    adv_embeddings = perturb_partial_sequence(embeddings, self.adv_epsilon, infos["clue_poss"])
                else:
                    adv_embeddings = perturb_whole_sequence(embeddings, self.adv_epsilon)

                # 确保新的扰动张量需要梯度
                context_adv_embeddings = context_adv_embeddings.detach().requires_grad_(True)
                clue_adv_embeddings = clue_adv_embeddings.detach().requires_grad_(True)

                torch.cuda.empty_cache()

                chosen_logps, rejected_logps, gpt_loss = self.concatenated_forward(
                    self.model,
                    inputs, attention_mask, context_adv_embeddings, 
                    inputs, attention_mask, clue_adv_embeddings, 
                    infos["input_length"], prompt_id_lens
                )

                losses, chosen_reward, reject_reward, _, _ = self.simpo_loss_fn(chosen_logps, rejected_logps)

                preference_loss = losses.mean()

                loss = preference_loss * self.gan_weight + gpt_loss * self.sft_weight
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                
                # print(f"overlap_ratio is: {overlap_ratio}")
                
                step_log_dict = {
                    "total_loss": loss.item(),
                    "chosen_reward": - chosen_reward.mean().item(),
                    "reject_reward": - reject_reward.mean().item(),
                    "overlap_ratio": overlap_ratio * dist.get_world_size(self.strategy.ring_attn_group),
                    "preference_loss": preference_loss.item(),
                    "adv_loss": adv_loss.item(),
                    "acc": acc,
                }

                # step bar
                step_log_dict = self.strategy.all_reduce(step_log_dict)
                if self._wandb is not None and self.strategy.is_rank_0():
                    logs = {"step_log/%s" % k: v for k, v in {**step_log_dict, "mini_batch_step": step}.items()}
                    self._wandb.log(logs)
                
                step_bar.set_postfix(step_log_dict)
                step_bar.update()


                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    logs_dict = {
                        "acc_mean": acc_mean,
                        "total_loss": loss_mean,
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    # step bar
                    logs_dict = self.strategy.all_reduce(logs_dict)
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                torch.cuda.empty_cache()

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
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # self.strategy.save_ckpt(
            #     self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            # )
            self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))


    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            short_ctx_loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompt_id_lens, inputs, attention_masks, infos, clue_prompt_id_lens, clue_inputs, clue_attention_masks in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                    clue_inputs = clue_inputs.to(torch.cuda.current_device()).squeeze(1)
                    clue_attention_mask = clue_attention_masks.to(torch.cuda.current_device()).squeeze(1)
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                    clue_output = self.model(
                        clue_inputs, 
                        attention_mask=clue_attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["clue_input_length"],
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
                    
                    short_ctx_loss_sum += short_ctx_gpt_loss.item()
                
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times, "eval short_ctx_gpt_loss": short_ctx_loss_sum / times}
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
    
    @torch.no_grad
    def compute_short_ctx_loss(self, clue_inputs, clue_attention_mask, infos, clue_prompt_id_lens):
        clue_output = self.model(
            clue_inputs, 
            attention_mask=clue_attention_mask, 
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=infos["clue_input_length"],
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
        return short_ctx_gpt_loss
    
    def concatenated_forward(self, model, chosen_ids, c_mask, c_embeddings, reject_ids, r_mask, r_embeddings, packed_seq_lens, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concat_inputs = torch.cat([chosen_ids, reject_ids], dim=-1)
        concat_embeddings = torch.cat([c_embeddings, r_embeddings], dim=1)
        concat_attention_mask = torch.cat([c_mask, r_mask], dim=1)
        
        # print(f"concat_inputs shape: {concat_inputs.shape}")
        # print(f"concat_embeddings shape: {concat_embeddings.shape}")
        # print(f"concat_attention_mask shape: {concat_attention_mask.shape}")
        # print(f"prompt_id_lens: {prompt_id_lens}")
        # print(f"packed_seq_lens: {packed_seq_lens}")

        output = model.forward_embedding(
            concat_inputs,
            inputs_embeds=concat_embeddings, 
            attention_mask=concat_attention_mask, 
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens * 2,
        )

        all_logits = output["logits"]
        # print(f"max logits: {all_logits.max()}")
        # print(f"min logits: {all_logits.min()}")

        all_logps = self._packed_get_batch_logps(
            all_logits, 
            concat_inputs, 
            concat_attention_mask, 
            prompt_id_lens * 2, 
            packed_seq_lens * 2,
            average_log_prob=True
        )

        # print(f"all_logps: {all_logps}")

        chosen_logps = all_logps[: 1]  # 这里默认的mini-bsz是2
        rejected_logps = all_logps[1 :] # 这里默认的mini-bsz是2
        
        return chosen_logps, rejected_logps, -all_logps[: 1].mean() # 这里默认的mini-bsz是2

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:

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
        index = 0
        # print(f"packed_seq_lens: {packed_seq_lens}")
        # print(f"seg_poss: {seg_poss}")
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len
        
        if average_log_prob:
            return torch.stack(logprobs_means)
        return torch.stack(logprobs_sums)