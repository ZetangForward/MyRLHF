#!/bin/bash

#SBATCH --job-name=llama_drop_3  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=48        # 添加CPU配置
#SBATCH --mem=400G                # 添加内存配置
#SBATCH --gres=gpu:4                       
#SBATCH --time=14-00:00:00       # 设置具体的时间限制，比如14天     
#SBATCH --output=/mnt/petrelfs/tangzecheng/MyRLHF/sbatch_logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/MyRLHF/sbatch_logs/%J.err         
#SBATCH --partition=belt_road

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

# 激活conda环境
conda activate zecheng_new

# 切换到工作目录
cd /mnt/petrelfs/tangzecheng/MyRLHF/build_data/preprocess_data/multi_hop

# 运行训练脚本
python cal_model_inference.py --drop_num=3 --model_path=meta-llama/Meta-Llama-3.1-8B-Instruct --out_file_path="/mnt/petrelfs/tangzecheng/local_data/processed_multi_hop/random_drop/llama" --num_gpus=8