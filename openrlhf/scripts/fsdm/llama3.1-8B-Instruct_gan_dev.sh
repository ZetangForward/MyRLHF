# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export HF_DATASETS_CACHE=/dev/null
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'
DATA_DIR='/mnt/petrelfs/tangzecheng/local_data'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/train_gan.py \
   --max_len 96000 \
   --dataset ${DATA_DIR}/processed_multi_hop/random_drop/train_llama_data/merge_v1_w_clues_dev \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/gan_dev \
   --ckpt_path ${SAVE_DIR}/merge_v1_fix/Llama-3.1-8B-Instruct/gan_dev \
   --save_steps 1 \
   --logging_steps 1 \
   --eval_steps 1 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "chosen" \
   --packing_samples \
   --bf16 \
   --adv_epsilon 0.05 \
   --sft_weight 0.5 \
   --beta 10 \
   --gamma_beta_ratio 0.3 \
   --num_processors 16 \
   --learning_rate 8e-7 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'merge_v1_full_gan_dev' \
   --wandb_run_name 'Llama-3.1-8B-Instruct-fsdm_dev' \
   --ring_attn_size 8 \
   --ring_head_stride 2;