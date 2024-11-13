#!/bin/bash

#SBATCH --job-name=train_llama_instruct    # 作业名
#SBATCH --nodes=1                          # 使用一个节点
#SBATCH --ntasks-per-node=1                # 每个节点的任务数
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=24                      # 每个节点需要8个GPU
#SBATCH --cpus-per-task=4                 # 分配给每个任务的CPU数目为该节点的CPU总数
#SBATCH --mem=448G                         # 使用该节点上所有可用内存
#SBATCH --time=infinite                    # 无限运行时间
#SBATCH --output=sbatch/logs/zc_job_id-%J.out        # 标准输出重定向到job_id-<jobid>.out文件
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         # 标准错误重定向到job_id-<jobid>.err文件
#SBATCH --partition=GPU_Monopolize2                    # 指定分区
#SBATCH --exclusive                        # 独占申请的节点

source activate zecheng

cd /public/home/zecheng/workspace/zecheng/MyRLHF/
bash scripts/train_sft_llama_dlcs.sh

