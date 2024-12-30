# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_sft.py \
   --max_len 96000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_qwen_data/merge_v1' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'Qwen/Qwen2.5-7B-Instruct' \
   --save_path ${SAVE_DIR}/merge_v1_fix/Qwen2.5-7B-Instruct/sft \
   --ckpt_path ${SAVE_DIR}/merge_v1_fix/Qwen2.5-7B-Instruct/sft \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_ckpt_num 5 \
   --max_epochs 1 \
   --input_key "chosen" \
   --packing_samples \
   --bf16 \
   --num_processors 16 \
   --learning_rate 1e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'merge_v1_fix' \
   --wandb_run_name 'Qwen2.5-7B-Instruct-sft' \
   --ring_attn_size 2;