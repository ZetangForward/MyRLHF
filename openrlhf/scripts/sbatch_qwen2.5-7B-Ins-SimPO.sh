#!/bin/bash

#SBATCH --job-name=SimPO-Qwen2.5-7B-Train
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=8 
#SBATCH --cpus-per-task=16        # 添加CPU配置
#SBATCH --mem=600G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=/mnt/petrelfs/tangzecheng/sbatch_logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/sbatch_logs/%J.err         
#SBATCH --partition=belt_road        
#SBATCH --quotatype=spot    
#SBATCH --exclusive     

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/MyRLHF/openrlhf

bash scripts/qwen2.5-7B-Instruct_simpo_pjlab.sh

# bash gen_offline.sh babilong reasoning meta-llama/Meta-Llama-3.1-8B-Instruct "RMT-team/babilong" /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct babilong 8 2