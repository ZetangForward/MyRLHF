BASE_DIR="/data/zecheng"
SAVE_DIR=/nvme/zecheng/ckpt
DATA_DIR=/nvme/zecheng/data/iclr2025/llama3-80k-train-data/dpo_data/chunk_16_size_1024
MODEL_DIR=${BASE_DIR}/hf_models

CUDA_VISIBLE_DEVICES=0,1 deepspeed --include localhost:0,1 openrlhf/cli/train_dpo_dev.py \
   --save_path /data/zecheng/MyRLHF/ckpt/llama3-8b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 1 \
   --micro_train_batch_size 1 \
   --pretrain ${MODEL_DIR}/'Llama-3-8B-Instruct-80K-QLoRA-Merged' \
   --bf16 \
   --max_epochs 1 \
   --max_len 10000 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset /nvme/zecheng/data/iclr2025/llama3-80k-train-data/dpo_data/chunk_16_size_1024 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --packing_samples \
   --flash_attn \
   --load_checkpoint \
   --deepspeed /data/zecheng/Retrieval_Head/iclr2025/training/config/zero3_offload.json \
   --gradient_checkpointing;