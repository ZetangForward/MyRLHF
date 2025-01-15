# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export HF_DATASETS_CACHE=/dev/null
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'
DATA_DIR='/mnt/petrelfs/tangzecheng/local_data'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_gan_nce.py \
   --max_len 80000 \
   --dataset ${DATA_DIR}/processed_multi_hop/random_drop/train_llama_data/merge_v2_w_clues \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --lora_rank 8 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/gan_nce \
   --ckpt_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/gan_nce \
   --save_steps 25 \
   --logging_steps 1 \
   --eval_steps 25 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "chosen" \
   --packing_samples \
   --bf16 \
   --adv_epsilon 0.05 \
   --sft_weight 1 \
   --nce_weight 0.0 \
   --num_processors 16 \
   --learning_rate 5e-7 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'merge_v1_fix_gan-full' \
   --wandb_run_name 'Llama-3.1-8B-Instruct-fsdm-nce0.5' \
   --ring_attn_size 8 \
   --ring_head_stride 4;