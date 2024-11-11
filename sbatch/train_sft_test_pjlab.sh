#!/bin/bash

#SBATCH --job-name=zecheng_test   
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:8                       
#SBATCH --time=infinite                  
#SBATCH --output=sbatch/logs/zc_job_id-%J.out       
#SBATCH --error=sbatch/logs/zc_job_id-%J.err         
#SBATCH --partition=belt_road             
#SBATCH --exclusive

source activate
conda activate zecheng

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

cd /mnt/petrelfs/tangzecheng/MyRLHF/

bash scripts/train_sft_llama3_pjlab.sh