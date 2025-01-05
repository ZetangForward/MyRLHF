python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/petrelfs/tangzecheng/local_ckpt/babilong/Llama-3.1-8B-Instruct/gan/convert_step200' \
    --tag global_step200_full \
    --adapter_path '' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/gan/Llama-3.1-8B-Instruct/' \
    --num_gpus 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --test_full \
    --tp_size 1 \


# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location