export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:0,1,2,3,4,5,6,7 openrlhf/cli/train_sft_dev.py \
   --max_len 64000 \
   --dataset /data/zecheng/acl2025/lcm_stack/dataset/training_data/dev_Qwen_query_answer_gen \
   --input_key instruction_str \
   --output_key pred_str \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain '/data/zecheng/hf_models/Qwen2-7B-Instruct' \
   --save_path './checkpoint/model/Qwen2-7B-Instruct-sft/' \
   --ckpt_path './checkpoint/opt/Qwen2-7B-Instruct-sft/' \
   --save_steps 20 \
   --num_process 48 \
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
   --wandb_run_name='Qwen2-7B-Instruct-sft-ring-4' \
   --ring_attn_size 4;