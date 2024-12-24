export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

deepspeed --include localhost:0,1 cli/train_simpo.py \
   --max_len 96000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_data/merge_v1' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/checkpoint/model/Llama-3-8B-Instruct-128k-tool-sft \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='openrlhf_simpo' \
   --wandb_run_name='dev' \
   --ring_attn_size 2