python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/petrelfs/tangzecheng/local_ckpt/global_step200' \
    --tag '' \
    --adapter_path '' \
    --save_path './babilong/contextual_denoise/Llama-3.1-8B-cd-lm_full/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;