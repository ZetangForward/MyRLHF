export CUDA_LAUNCH_BLOCKING=1
SAVE_DIR='/public/home/zecheng/workspace/zecheng/ckpt/acl2025'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:0,1,2,3,4,5,6,7 openrlhf/cli/train_sft_dev.py \
   --max_len 64000 \
   --dataset /public/home/zecheng/workspace/zecheng/data/acl2025/Qwen_query_answer_gen \
   --input_key instruction_str \
   --output_key pred_str \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain '/public/home/zecheng/workspace/hf_models/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/checkpoint/model/llama3.1-8b-tool-sft \
   --ckpt_path ${SAVE_DIR}/checkpoint/opt/llama3.1-8b-tool-sft \
   --save_steps 50 \
   --num_process 20 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 3 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_tensorboard ${SAVE_DIR}/checkpoint/tensorboard/llama3.1-8b-tool-sft/tensorboard \
   --disable_fast_tokenizer \
   --ring_attn_size 8;