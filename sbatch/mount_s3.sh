#!/bin/bash
#SBATCH --job-name=mount_job            # 作业名称
#SBATCH --partition=belt_road           # 分区名称
#SBATCH --gres=gpu:0                    # 不需要GPU则设置为0
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=1              # 每个任务使用的CPU数量
#SBATCH --output=sbatch/logs/mount_job_id-%J.out       
#SBATCH --error=sbatch/logs/mount_job_id-%J.err   
#SBATCH --ntasks-per-node=1             # 每节点一个任务

# 切换到目标工作目录
cd /mnt/petrelfs/tangzecheng

# 确保目标目录存在
# rm -rf /nvme/chenguanjie/bucket
# rm -rf /nvme/chenguanjie/bucket_cache/mountpoint_cache

if [ ! -d "/mnt/petrelfs/tangzecheng/llm-data-exp-space-2" ]; then
    mkdir -p /mnt/petrelfs/tangzecheng/llm-data-exp-space-2
fi


# 确保目标目录存在
if [ ! -d "/mnt/petrelfs/tangzecheng/s3mount_cache" ]; then
    mkdir -p /mnt/petrelfs/tangzecheng/s3mount_cache/mountpoint_cache
fi

# 等待几秒钟，确保文件系统准备就绪
sleep 5

# 执行挂载命令

proxy_off

/mnt/petrelfs/tangzecheng/s3mount wulijun_blob /mnt/petrelfs/tangzecheng/llm-data-exp-space-2 --cache /mnt/petrelfs/tangzecheng/s3mount_cache --checkpoint-dir mountpoint_cache --endpoint-url http://p-ceph-hdd2-outside.pjlab.org.cn --force-path-style --allow-delete --allow-overwrite --debug --log-directory /mnt/petrelfs/tangzecheng/MyRLHF/mount_log --max-threads 32 


# 防止脚本退出，保持挂载
sleep infinity