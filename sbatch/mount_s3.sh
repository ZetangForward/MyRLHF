#!/bin/bash

# 检查是否已经挂载
if mount | grep -q '/mnt/petrelfs/tangzecheng/llm-data-exp-space'; then
    echo "S3已经挂载."
else
    echo "正在挂载S3..."
    ./s3mount wulijun_blob /mnt/petrelfs/tangzecheng/llm-data-exp-space --allow-delete --allow-overwrite --endpoint-url http://10.140.31.252:80
    echo "挂载完成."
fi
