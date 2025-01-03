python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --adapter_path '/mnt/petrelfs/tangzecheng/local_ckpt/babilong/Llama-3.1-8B-Instruct/sft/global_step200' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/merge_v1_fix/Llama-3.1-8B-Instruct/' \
    --num_gpus 6 \
    --tp_size 1 \


# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location