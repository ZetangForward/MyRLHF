BASE_DIR="/data/zecheng"
SAVE_DIR=/nvme/zecheng/ckpt
DATA_DIR=/nvme/zecheng/data/iclr2025/llama3-80k-train-data/dpo_data/chunk_16_size_1024
MODEL_DIR=${BASE_DIR}/hf_models

CUDA_VISIBLE_DEVICES=0,1 deepspeed --include localhost:0,1 openrlhf/cli/train_sft_dev.py \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain ${MODEL_DIR}/'Llama-3-8B-Instruct-80K-QLoRA-Merged' \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing

