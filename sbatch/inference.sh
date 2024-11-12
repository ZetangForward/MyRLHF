#!/bin/bash

#SBATCH --job-name=zecheng_test   
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=48        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road             
#SBATCH --exclusive

# 确保日志目录存在
mkdir -p sbatch/logs

# 设置错误处理
set -e  # 遇到错误立即退出
set -x  # 打印执行的命令

# 激活conda环境
source activate
conda activate zecheng_new

# 设置代理
export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

# 切换到工作目录
cd /mnt/petrelfs/tangzecheng/MyRLHF/inference/llama3-1-8B

# 打印一些环境信息方便调试
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 运行训练脚本
bash gen_offline.sh rapid_single_api tool_calling