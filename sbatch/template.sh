#!/bin/bash

#SBATCH --job-name=auto_amount_template  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=48        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road       

mkdir -p sbatch/logs

echo "Running on nodes: $SLURM_NODELIST"

# 使用 $SLURM_NODELIST 作为节点列表
srun -p belt_road -w $SLURM_NODELIST sbatch/mount_s3.sh

echo "挂载完毕，准备执行程序"

sleep 5

source ~/anaconda3/bin/activate zecheng_new

bash scripts/train_sft_llama_ring.sh

sleep infinity