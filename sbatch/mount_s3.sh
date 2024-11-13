#!/bin/bash
#SBATCH --job-name=mount_job            # 作业名称
#SBATCH --partition=belt_road           # 分区名称
#SBATCH --gres=gpu:0                    # 不需要GPU则设置为0
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=1              # 每个任务使用的CPU数量
#SBATCH --output=sbatch/logs/mount_job_id-%J.out       
#SBATCH --error=sbatch/logs/mount_job_id-%J.err   
#SBATCH --ntasks-per-node=1             # 每节点一个任务

echo "取消上次挂载"
fusermount -u /mnt/petrelfs/tangzecheng/llm-data-exp-space-2

if [ ! -d "/mnt/petrelfs/tangzecheng/llm-data-exp-space-2" ]; then
    mkdir -p /mnt/petrelfs/tangzecheng/llm-data-exp-space-2
fi


# 等待几秒钟，确保文件系统准备就绪
sleep 5

# 执行挂载命令
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

echo "挂载磁盘"
/mnt/petrelfs/tangzecheng/s3mount wulijun_blob /mnt/petrelfs/tangzecheng/llm-data-exp-space-2 --allow-delete --allow-overwrite --endpoint-url http://10.140.31.252:80

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/