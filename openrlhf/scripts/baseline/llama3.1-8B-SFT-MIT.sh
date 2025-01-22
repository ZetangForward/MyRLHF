# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --master_port 29504 --include localhost:0,1,2,3,4,5,6,7 cli/train_sft.py \
   --max_len 96000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_llama_data/merge_v1_w_clues' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain 'Crystalcareai/meta-llama-3.1-8b' \
   --save_path ${SAVE_DIR}/baseline/Llama-3.1-8B/merge_v1_fix_sft \
   --ckpt_path ${SAVE_DIR}/baseline/Llama-3.1-8B/merge_v1_fix_sft  \
   --input_template "User: {}\nAssistant: " \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
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
   --wandb_project 'baseline' \
   --wandb_run_name 'Llama-3.1-8B-LongMIT-sft' \
   --ring_attn_size 4 \
   --ring_head_stride 2;