python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --tag dpo_gan_v2_step125 \
    --adapter_path '/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/gan_sft_v2/global_step125' \
    --save_path '/mnt/petrelfs/tangzecheng/MyRLHF/inference/babilong/gan/Llama-3.1-8B-Instruct/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;


# bash gen.sh rapid_parallel_api tool_calling
# bash gen.sh rapid_multiple_api tool_calling
# bash gen.sh rapid_single_api tool_calling
# bash gen.sh rapid_parallel_api tool_location
# bash gen.sh rapid_multiple_api tool_location
# bash gen.sh rapid_single_api tool_location