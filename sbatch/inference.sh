#!/bin/bash

#SBATCH --job-name=zecheng_test   
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=24        # 添加CPU配置
#SBATCH --mem=200G                # 添加内存配置
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road             
#SBATCH --exclusive

# 激活别名支持
shopt -s expand_aliases

# 激活conda环境
source ~/.bashrc
conda activate zecheng_new

proxy_off 

cd ~

rm -rf /mnt/petrelfs/tangzecheng/llm-data-exp-space
mkdir -p '/mnt/petrelfs/tangzecheng/llm-data-exp-space'

# 检查是否已经挂载
if mount | grep -q '/mnt/petrelfs/tangzecheng/llm-data-exp-space'; then
    echo "S3已经挂载."
else
    echo "正在挂载S3..."
    ./s3mount wulijun_blob /mnt/petrelfs/tangzecheng/llm-data-exp-space --allow-delete --allow-overwrite --endpoint-url http://10.140.31.252:80
    echo "挂载完成."
fi

proxy_on

# 切换到工作目录
cd /mnt/petrelfs/tangzecheng/MyRLHF/inference/llama3-1-8B

# 打印一些环境信息方便调试
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 运行训练脚本
bash gen_offline.sh rapid_multiple_api tool_calling