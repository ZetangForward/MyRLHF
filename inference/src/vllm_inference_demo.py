import argparse
import json
import multiprocessing as mp
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def worker(gpu_id, prompts_chunk, original_data_chunk, args, return_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=2048, n=args.k)

    outputs = llm.generate(prompts_chunk, sampling_params=sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generations = [o.text for o in output.outputs]
        original_prompt_data = original_data_chunk[i]  # 获取原始数据
        original_prompt_data['responses'] = generations  # 插入生成的响应
        results.append(original_prompt_data)

    return_list.extend(results)

def main():
    parser = argparse.ArgumentParser(description="使用 vllm 对 prompts 进行生成（数据并行）")
    parser.add_argument('--jsonl_path', type=str, required=True, help='输入的 jsonl 文件路径，每一行是一个包含 prompt 的字典')
    parser.add_argument('--out_path', type=str, required=True, help='生成结果的输出路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成的温度参数')
    parser.add_argument('--k', type=int, default=1, help='每个 prompt 生成的数量 K')
    parser.add_argument('--num_gpus', type=int, default=8, help='使用的 GPU 数量')
    args = parser.parse_args()

    parent_dir = os.path.dirname(args.out_path)
    os.makedirs(parent_dir, exist_ok=True)
    
    # 读取输入的 jsonl 文件
    prompts = []
    original_data = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
            original_data.append(data)

    # 将 prompts 平均分配到多个 GPU
    num_gpus = args.num_gpus
    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    prompts_chunks = [prompts[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    original_data_chunks = [original_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]

    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    # 使用 tqdm 显示总进度
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, prompts_chunks[gpu_id], original_data_chunks[gpu_id], args, return_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
           

    # 保存生成结果
    with open(args.out_path, 'w', encoding='utf-8') as f_out:
        for result in return_list:
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
