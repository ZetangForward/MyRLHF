export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_simpo.py \
   --max_len 80000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_data/merge_v1' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/simpo \
   --ckpt_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/simpo \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 3 \
   --nll_loss_coef 0.1 \
   --max_ckpt_num 5 \
   --max_epochs 1 \
   --packing_samples \
   --bf16 \
   --beta 10 \
   --gamma_beta_ratio 0.3 \
   --num_processors 14 \
   --learning_rate 1e-6 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='merge_v1_fix' \
   --wandb_run_name='Llama-3.1-8B-Instruct-SimPO-beta-10-gamma_beta_ratio-0.3-nll_loss_coef-0.1' \
   --ring_attn_size 4