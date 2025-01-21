python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-ce/global_step150_hf' \
    --tag 'ce_200' \
    --adapter_path '' \
    --save_path './babilong/baseline/Llama-3.1-8B-ce_200/' \
    --gpu_id 4,5,6,7 \
    --tp_size 1;


# nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 bash Llama-3.1-8B-Base/gen_ce.sh > ./logs/gen_ce.log 2>&1 &