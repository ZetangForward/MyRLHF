python inference_babilong.py \
    --dataset_name "RMT-team/babilong" \
    --model_path '/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3.1-8B-pg19-longce/200ep' \
    --tag 'long_ce_200' \
    --adapter_path '' \
    --save_path './babilong/baseline/Llama-3.1-8B-long_ce_200/' \
    --gpu_id 0,1,2,3 \
    --tp_size 1;


# nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 bash Llama-3.1-8B-Base/gen_longce.sh > ./logs/gen_longce.log 2>&1 &