#!/bin/bash

# 检查是否提供了命令参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

# 读取命令参数
COMMAND="$1"

# 创建临时脚本文件
TMP_SCRIPT="sbatch/inference_tmp.sh"
cp sbatch/inference_template.sh "$TMP_SCRIPT"

# 将命令追加到临时脚本
echo "bash gen_offline.sh $COMMAND" >> "$TMP_SCRIPT"

# 提交 sbatch 任务并捕获作业 ID
job_id=$(sbatch "$TMP_SCRIPT" | grep -o '[0-9]\+')

# 打印作业信息
echo "Submitted job $job_id with command: bash gen_offline.sh $COMMAND"

# 删除临时脚本
rm "$TMP_SCRIPT"
