import os
from abc import ABC
import torch
from typing import Dict, List, Optional, Tuple, Union
import torch.distributed as dist
from flash_attn.utils.distributed import all_gather
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm
from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class FDSMTrainerV2(ABC):
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
        opposite: bool = False,
        loss_weight: float = 1.0, 
        w_annotation: bool = False,
        perturb_type: str = None
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
        self.w_annotation = w_annotation
        self.perturb_type = perturb_type
        self.opposite = opposite
        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group, loss_weight=loss_weight)

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

    def adjust_weights_by_gradient(self, sequence, epsilon):
        """
        Reduce weights in the sequence for regions with smaller gradients.

        Args:
            sequence (torch.Tensor): The sequence to adjust, shape (b, l, d).
            epsilon (float): Scaling factor to control the adjustment strength.

        Returns:
            torch.Tensor: Adjusted sequence with reduced weights for smaller gradients.
        """
        dist.all_reduce(sequence.grad, op=dist.ReduceOp.SUM, group=self.strategy.ring_attn_group)
        
        # Compute L2 norm of the gradient for each position
        grad_l2_norm = torch.norm(sequence.grad, p=2, dim=2)  # (b, l)
        
        # Normalize the gradient to [0, 1] (optional, to stabilize scaling)
        grad_l2_norm_normalized = grad_l2_norm / (grad_l2_norm.max(dim=-1, keepdim=True)[0] + 1e-8)  # (b, l)
        
        # Compute scaling factor: Smaller gradients -> Smaller scaling values
        scaling_factor = 1 - epsilon * (1 - grad_l2_norm_normalized)  # (b, l)
        
        # Apply scaling to the sequence
        adjusted_sequence = sequence * scaling_factor.unsqueeze(-1)  # (b, l, d)
        
        return adjusted_sequence


    def perturb_ipt_embedding(self, sequence, epsilon=None, return_pos=False, opposite=False):
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
        dist.all_reduce(sequence.grad, op=dist.ReduceOp.SUM, group=self.strategy.ring_attn_group)

        # Calculate the L2 norm of the gradient for each position in the sequence
        grad_l2_norm = torch.norm(sequence.grad, p=2, dim=2)  # (b, l)
        avg_grad_l2_norm = grad_l2_norm.mean(dim=-1, keepdim=True)  # (b, 1)
        
        # Create a mask indicating whether each position's gradient is larger than the average
        is_large_gradient = grad_l2_norm > avg_grad_l2_norm  # (b, l)
        
        if return_pos:
            return sequence, is_large_gradient
        
        sigma = epsilon / (1 + grad_l2_norm)  # (b, l)
        # Generate Gaussian noise with the same shape as the sequence
        # noise = torch.randn_like(sequence, device=sequence.device)  # (b, l, d)
        # noise = noise * grad_sign  # (b, l, d)
        
        # Apply perturbation in the direction based on gradient size
        # For large gradients, perturb in the opposite direction
        # For small or equal gradients, perturb in the same direction
        if opposite:
            perturbation = -sigma.unsqueeze(-1) * sequence.grad * ~is_large_gradient.unsqueeze(-1)
        else:
            perturbation = -sigma.unsqueeze(-1) * sequence.grad * is_large_gradient.unsqueeze(-1)
        # perturbation = torch.where(
        #     is_large_gradient.unsqueeze(-1),  # (b, l, 1)
        #     -sigma.unsqueeze(-1) * sequence.grad,    # Opposite direction for large gradients
        #     0
        #     # sigma.unsqueeze(-1) * sequence.grad      # Same direction for small gradients
        # )

        return sequence + perturbation, None

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
            loss_mean = 0
            for data in self.train_dataloader:
                prompt_id_lens, inputs, attention_masks, infos, clue_prompt_id_lens, clue_inputs, clue_attention_masks = data
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

                if self.args.lora_rank != 0:
                    self.model.model.base_model.disable_adapter_layers()
                    embedding_layer = self.model.model.base_model.model.get_input_embeddings()
                else:
                    embedding_layer = self.model.model.get_input_embeddings()

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

                for param in self.model.model.parameters():
                    param.requires_grad = True

                if self.args.lora_rank != 0:
                    self.model.model.base_model.enable_adapter_layers()

                if False:  # FIXME: we do not take this into consideration
                    adv_embeddings = self.adjust_weights_by_gradient(embeddings, self.adv_epsilon)                    
                elif self.perturb_type == "embedding":
                    adv_embeddings, perturb_pos = self.perturb_ipt_embedding(embeddings, self.adv_epsilon, return_pos=False, opposite=self.opposite)
                elif self.perturb_type == "loss":
                    adv_embeddings, perturb_pos = self.perturb_ipt_embedding(embeddings, return_pos=True)
                # detach from the original gradient graph
                adv_embeddings = adv_embeddings.detach().requires_grad_(True)

                torch.cuda.empty_cache()

                output = self.model.forward_embedding(
                    inputs,
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["input_length"],
                )

                gpt_loss = self.loss_fn(output.logits, labels, perturb_pos)
                self.strategy.backward(gpt_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }

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
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompt_id_lens, inputs, attention_masks, infos, _, _, _ in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
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