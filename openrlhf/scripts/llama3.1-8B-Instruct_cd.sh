# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export HF_DATASETS_CACHE=/dev/null
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_cd.py \
   --max_len 96000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_llama_data/merge_v1_w_clues_dev' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/cd \
   --ckpt_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/cd \
   --save_steps 25 \
   --logging_steps 1 \
   --eval_steps 25 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "chosen" \
   --packing_samples \
   --bf16 \
   --target_modules "q_proj" "k_proj" \
   --num_processors 16 \
   --learning_rate 1e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'merge_v1_cd' \
   --wandb_run_name 'Llama-3.1-8B-Instruct' \
   --ring_attn_size 4 \
   --ring_head_stride 2;