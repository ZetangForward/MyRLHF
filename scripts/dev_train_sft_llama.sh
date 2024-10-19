BASE_DIR="/data/zecheng"
SAVE_DIR=/nvme/zecheng/ckpt
DATA_DIR=/nvme/zecheng/data/iclr2025/llama3-80k-train-data/dpo_data/chunk_16_size_1024
MODEL_DIR=${BASE_DIR}/hf_models

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:0,1,2,3,4,5,6,7 openrlhf/cli/train_sft_dev.py \
   --max_len 48000 \
   --dataset /data/zecheng/lcm_stack/dataset/training_data/Qwen_query_answer_gen.jsonl \
   --input_key instruction_str \
   --output_key pred_str \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --lora_rank 32 \
   --apply_chat_template \
   --max_samples 500000 \
   --pretrain '/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct' \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 3 \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard \
   --packing_samples;

