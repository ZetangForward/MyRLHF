python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/pg19/Llama-3.1-8B/cd_lm_full-0.005/global_step300' \
    --tag '/cd_lm_full-0.005_300' \
    --adapter_path '' \
    --save_path './babilong/contextual_denoise/Llama-3.1-8B-/cd_lm_full-0.005/' \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --tp_size 1;


# nohup bash Llama-3.1-8B-Base/gen_cd_full.sh > ./logs/gen_cd_full.log 2>&1 &