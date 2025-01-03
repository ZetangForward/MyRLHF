export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:0,1,2,3,4,5,6,7 openrlhf/cli/train_sft_dev.py \
   --max_len 64000 \
   --dataset /data/zecheng/lcm_stack/dataset/training_data/dev_Qwen_query_answer_gen \
   --input_key instruction_str \
   --output_key pred_str \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain '/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct' \
   --save_path './checkpoint/model/llama3.1-8b-sft/' \
   --ckpt_path './checkpoint/opt/llama3.1-8b-sft/' \
   --save_steps 20 \
   --num_process 96 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 5 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='debug_openrlhf_train_sft' \
   --wandb_run_name='llama3.1-8b-sft-vanilla';