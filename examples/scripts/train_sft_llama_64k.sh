export CUDA_LAUNCH_BLOCKING=1
CKPT_PATH='/mnt/petrelfs/tangzecheng/llm-data-exp-space-2/zecheng/remote_ckpt'

deepspeed --include localhost:0 openrlhf/cli/train_sft_dev.py \
   --max_len 64000 \
   --dataset "/mnt/petrelfs/tangzecheng/long-context-gpt-build-data/gpt" \
   --input_key "train" \
   --output_key "" \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3-8B-Instruct' \
   --save_path ${CKPT_PATH}/model/llama3-8b-sft \
   --ckpt_path ${CKPT_PATH}/model/llama3-8b-sft \
   --save_steps 15 \
   --num_process 2 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 5 \
   --bf16 \
   --pretrain_mode \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='debug_openrlhf_train_sft' \
   --wandb_run_name='llama3-8b-sft-vanilla' \