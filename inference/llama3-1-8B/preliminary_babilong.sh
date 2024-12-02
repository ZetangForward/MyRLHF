export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

data_name=$1
task_name=$2
model_path=$3
benchmark_dir=$4
out_dir=$5
benchmark_name=$6
num_gpus=$7
dialogue_turn=$8

mkdir -p $out_dir

python preliminary_babilong.py \
    --model_path $model_path \
    --data_dir $benchmark_dir \
    --save_path ${out_dir}/${data_name}/${task_name} \
    --task_name $task_name \
    --dataset_name $data_name \
    --benchmark_name $benchmark_name \
    --max_workers 24 \
    --max_model_len 128000 \
    --tp_size 2 \
    --dialogue_turn $dialogue_turn \
    --num_gpus $num_gpus;


# bash preliminary_babilong.sh babilong reasoning meta-llama/Meta-Llama-3.1-8B-Instruct /mnt/petrelfs/tangzecheng/Long-form-reasoning-data/data/generated_tasks /mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/preliminary/prompt_engineering babilong 8 1

# bash preliminary_babilong.sh babilong reasoning /data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct /data/zecheng/Long-form-reasoning-data/data/generated_tasks /data/zecheng/acl2025/MyRLHF/evaluation/babilong/llama-3_1-8B-Instruct babilong 6 1

