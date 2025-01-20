#!/bin/bash

#SBATCH --job-name=11900_reasoning_head_detection  # 任务名称
#SBATCH --nodes=1                            # 使用1个节点
#SBATCH --ntasks-per-node=1                  # 每个节点运行1个任务
#SBATCH --cpus-per-task=8                    # 每个任务使用8个CPU核心
#SBATCH --mem=64G                            # 内存配置
#SBATCH --gres=gpu:1                         # 使用1张GPU卡
#SBATCH --time=14-00:00:00                   # 时间限制为14天
#SBATCH --output=/mnt/petrelfs/tangzecheng/MyRLHF/reetrievalheaddetect/logs/11900.log  # 标准输出日志
#SBATCH --error=/mnt/petrelfs/tangzecheng/MyRLHF/reetrievalheaddetect/logs/11900.err   # 错误日志
#SBATCH --partition=belt_road                # 指定分区

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source ~/.bashrc

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/MyRLHF/reetrievalheaddetect

python reasoning_head_detection_babilong.py --context_lengths=11900