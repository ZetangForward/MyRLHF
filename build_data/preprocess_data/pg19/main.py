from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

# 加载tokenizer和数据集
data = load_dataset("/mnt/petrelfs/tangzecheng/local_data/pg19", trust_remote_code=True)
max_sequence_length = 32000

train_data, validation_data = data['train'], data['validation']

def process_text(text, max_seq_length, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    results = []
    if len(tokens) > max_seq_length:
        # 将过长的文本分成多个段
        num_segments = math.ceil(len(tokens) / max_seq_length)
        for i in range(num_segments):
            start = i * max_seq_length
            end = (i + 1) * max_seq_length
            segment = tokens[start:end]
            text_segment = tokenizer.decode(segment, skip_special_tokens=True)
            results.append({'text': text_segment})
    else:
        # 如果长度不足，直接返回
        text_segment = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append({'text': text_segment})
    return results

def process_subset(subset, max_seq_length):
    results = []
    tokenizer = AutoTokenizer.from_pretrained("Crystalcareai/meta-llama-3.1-8b")
    for text in subset:
        results.extend(process_text(text, max_seq_length, tokenizer))
    return results

def process_dataset(dataset, max_seq_length, num_proc=None):
    if num_proc is None:
        num_proc = cpu_count()  # 默认使用所有CPU核心
    
    # 将数据集分成多个子集
    texts = dataset['text']
    subset_size = len(texts) // num_proc
    subsets = [texts[i * subset_size:(i + 1) * subset_size] for i in range(num_proc)]
    
    # 使用多进程处理
    with Pool(num_proc) as pool:
        results = list(tqdm(
            pool.starmap(process_subset, [(subset, max_seq_length) for subset in subsets]),
            total=num_proc,
            desc="Processing dataset"
        ))
    
    # 合并所有结果
    final_results = []
    for result in results:
        final_results.extend(result)
    
    # 创建新的Dataset
    new_dataset = Dataset.from_list(final_results)
    return new_dataset

# 处理训练集和验证集
print("Processing training data...")
train_data = process_dataset(train_data, max_sequence_length, num_proc=48)

import pdb; pdb.set_trace()

print("Processing validation data...")
validation_data = process_dataset(validation_data, max_sequence_length, num_proc=48)

# 查看处理后的数据
print(train_data[0])
print(validation_data[0])