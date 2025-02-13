#!/bin/bash

#SBATCH --job-name=zc_debugger    # 作业名
#SBATCH --nodes=1                          # 使用一个节点
#SBATCH --ntasks-per-node=1                # 每个节点的任务数
#SBATCH --gres=gpu:4                       # 每个节点需要8个GPU
#SBATCH --cpus-per-task=20                 # 分配给每个任务的CPU数目为该节点的CPU总数
#SBATCH --mem=400G                         # 使用该节点上所有可用内存
#SBATCH --time=infinite                    # 无限运行时间
#SBATCH --output=sbatch/logs/zc_job_id-%J.out        # 标准输出重定向到job_id-<jobid>.out文件
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         # 标准错误重定向到job_id-<jobid>.err文件
#SBATCH --partition=GPU_Monopolize2                    # 指定分区

source activate zecheng

cd /public/home/zecheng/workspace/zecheng/MyRLHF/
bash scripts/train_sft_phi_dlcs.sh