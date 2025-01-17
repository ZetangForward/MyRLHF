# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/analysis_train_process.py \
   --max_len 65000 \
   --lora_rank 32 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/analysis_multihop_data/babilong_qa3' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/baseline/Llama-3.1-8B-Instruct/analysis/babilong_qa3 \
   --ckpt_path ${SAVE_DIR}/baseline/Llama-3.1-8B-Instruct/analysis/babilong_qa3 \
   --save_steps 25 \
   --logging_steps 1 \
   --eval_steps 1000000000 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "message" \
   --bf16 \
   --num_processors 16 \
   --learning_rate 1e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'analysis' \
   --wandb_run_name 'Llama-3.1-8B-Instruct-babilong-qa3';