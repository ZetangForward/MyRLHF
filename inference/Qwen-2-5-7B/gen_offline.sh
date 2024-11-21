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

mkdir -p $out_dir

python inference.py \
    --model_path $model_path \
    --data_dir $benchmark_dir \
    --save_path ${out_dir}/${data_name}/${task_name} \
    --task_name $task_name \
    --dataset_name $data_name \
    --benchmark_name $benchmark_name \
    --max_workers 24 \
    --max_model_len 128000 \
    --tp_size 2 \
    --num_gpus $num_gpus;

# bash gen_offline.sh rapid_multiple_api tool_location Qwen/Qwen2.5-7B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/Qwen-2-5-7b-instruct api

# bash gen_offline.sh rapid_parallel_api tool_location Qwen/Qwen2.5-7B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/Qwen-2-5-7b-instruct api

# bash gen_offline.sh rapid_single_api tool_location meta-llama/Qwen/Qwen2.5-7B-Instruct /mnt/hwfile/opendatalab/tangzecheng/benchmark_data /mnt/petrelfs/tangzecheng/local_data/inference_results/Qwen-2-5-7b-instruct api