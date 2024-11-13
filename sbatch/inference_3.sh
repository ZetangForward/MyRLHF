#!/bin/bash

#SBATCH --job-name=inference_3  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=24        # 添加CPU配置
#SBATCH --mem=200G                # 添加内存配置
#SBATCH --gres=gpu:4                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

# 激活conda环境
conda activate zecheng_new

# 切换到工作目录
cd /mnt/petrelfs/tangzecheng/MyRLHF/inference/llama3-1-8B

# 运行训练脚本
bash gen_offline.sh rapid_parallel_api tool_calling