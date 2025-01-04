python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/petrelfs/tangzecheng/local_ckpt/babilong/Llama-3.1-8B-Instruct/gan/convert_step200' \
    --tag global_step200 \
    --adapter_path '' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/gan/Llama-3.1-8B-Instruct/' \
    --num_gpus 4 \
    --tp_size 1 \


# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location