export CUDA_VISIBLE_DEVICES=2

python retrieval_head_detection.py \
    --model_path "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct" \
    --multi_hop_reasoning;
