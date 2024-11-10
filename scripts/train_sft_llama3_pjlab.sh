MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
echo $MASTER_ADDR
echo $MASTER_PORT

SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

deepspeed --hostfile=hostfile --num_nodes 1 --no_ssh_check --master_addr $MASTER_ADDR --launcher SLURM openrlhf/cli/train_sft_dev.py \
   --max_len 64000 \
   --dataset '/mnt/petrelfs/tangzecheng/transfer_data/Qwen_query_answer_gen' \
   --input_key instruction_str \
   --output_key pred_str \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --apply_chat_template \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
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
   --wandb_run_name='Llama-3-8B-Instruct-80K-tool-sft-ring-2' \
   --ring_attn_size 2