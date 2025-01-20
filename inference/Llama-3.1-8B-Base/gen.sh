python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path 'Crystalcareai/meta-llama-3.1-8b' \
    --tag '' \
    --adapter_path '/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/merge_v1_fix/Llama-3.1-8B-Instruct/dev/global_step150' \
    --save_path './babilong/contextual_denoise/Llama-3.1-8B-cd-v1/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;