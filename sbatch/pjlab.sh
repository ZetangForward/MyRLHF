#!/bin/bash
#SBATCH -J test
#SBATCH -p belt_road
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:8
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road   

source activate zecheng_new

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
echo $MASTER_ADDR
echo $MASTER_PORT

hostfile=""
deepspeed --hostfile=$hostfile --launcher SLURM openrlhf/cli/train_sft_dev.py \
    --max_len 64000 \
    --dataset '/mnt/petrelfs/tangzecheng/transfer_data/Qwen_query_answer_gen' \
    --input_key instruction_str \
    --output_key pred_str \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --lora_rank 32 \
    --apply_chat_template \
    --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --save_path /mnt/petrelfs/tangzecheng/local_ckpt/checkpoint/model/Llama-3-8B-Instruct-128k-tool-sft \
    --ckpt_path /mnt/petrelfs/tangzecheng/local_ckpt/checkpoint/opt/Llama-3-8B-Instruct-tool-sft \
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
    --wandb_run_name='Llama-3-8B-Instruct-80K-tool-sft-ring-2' \
    --ring_attn_size 2