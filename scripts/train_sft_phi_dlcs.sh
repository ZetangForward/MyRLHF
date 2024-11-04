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
   --pretrain '/public/home/zecheng/workspace/hf_models/aaronday3/Phi-3-small-128k-instruct' \
   --save_path ${SAVE_DIR}/checkpoint/model/Phi-3-small-128k-tool-sft \
   --ckpt_path ${SAVE_DIR}/checkpoint/opt/Phi-3-small-128k-tool-sft \
   --save_steps 50 \
   --num_process 20 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 1 \
   --max_epochs 1 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='openrlhf_sft' \
   --wandb_run_name='Phi-3-small-128k-tool-sft-ring-4' \
   --ring_attn_size 4;