# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_babilong.py \
   --max_len 48000 \
   --dataset 'RMT-team/babilong-train-5k-samples' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/babilong/Llama-3.1-8B-Instruct/sft_full \
   --ckpt_path ${SAVE_DIR}/babilong/Llama-3.1-8B-Instruct/sft_full \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_ckpt_num 5 \
   --max_epochs 2 \
   --input_key "chosen" \
   --packing_samples \
   --bf16 \
   --num_processors 16 \
   --learning_rate 8e-7 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'babilong' \
   --wandb_run_name 'Llama-3.1-8B-babilong-sft_full' \
   --ring_attn_size 2;