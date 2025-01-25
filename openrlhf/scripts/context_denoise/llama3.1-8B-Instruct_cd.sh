# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export HF_DATASETS_CACHE=/dev/null
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/train_sft.py \
   --max_len 96000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/train_llama_data/merge_v1_w_clues_dev' \
   --train_batch_size 4 \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --pretrain 'Crystalcareai/meta-llama-3.1-8b' \
   --save_path ${SAVE_DIR}/merge_v2/Llama-3.1-8B-Instruct/ins-opposite_gradient_large_pos-1e-3 \
   --ckpt_path ${SAVE_DIR}/merge_v2/Llama-3.1-8B-Instruct/ins-opposite_gradient_large_pos-1e-3 \
   --save_steps 1 \
   --logging_steps 1 \
   --eval_steps 1 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "chosen" \
   --adv_epsilon 1e-3 \
   --packing_samples \
   --input_template "User: {}\nAssistant: " \
   --bf16 \
   --num_processors 16 \
   --learning_rate 8e-7 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project '二代机' \
   --wandb_run_name 'ins-opposite_gradient_large_pos-1e-3' \
   --ring_attn_size 8 \
   --ring_head_stride 2;