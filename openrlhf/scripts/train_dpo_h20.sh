export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

SAVE_DIR='/data/zecheng/ckpt'

deepspeed --include localhost:6,7 openrlhf/cli/train_simpo.py \
   --max_len 96000 \
   --dataset '/data/zecheng/repos/check_inference/llama/inference_drop_1.pkl' \
   --prompt_key instruction_str \
   --chosen_key pred_str \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain '/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/checkpoint/model/Llama-3-8B-Instruct-128k-tool-sft \
   --ckpt_path ${SAVE_DIR}/checkpoint/opt/Llama-3-8B-Instruct-tool-sft \
   --save_steps 50 \
   --num_process 20 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb='f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project='openrlhf_sft' \
   --wandb_run_name='Llama-3-8B-Instruct-80K-tool-dpo-dev' \
   --ring_attn_size 2