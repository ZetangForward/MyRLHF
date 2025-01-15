from datasets import load_dataset
import json
import os
from transformers import AutoTokenizer
from modelzipper.tutils import *
import itertools

def calculate_clue_length(clue_docs, tokenizer):
    total_length = 0
    for doc in clue_docs:
        total_length += len(tokenizer(doc['content'], add_special_tokens=False)['input_ids'])
    return total_length


def calculate_clue_length(clue_docs, tokenizer):
    # 计算 clue_docs 的总长度
    total_length = 0
    for doc in clue_docs:
        total_length += len(tokenizer(doc['content'], add_special_tokens=False)['input_ids'])
    return total_length

def process_longmit_datasets(
    dataset_name: str = "/data/zecheng/data/LongMIT-128K", 
    save_dir: str = None,
    language: str = "en",
    max_length: int = 128000,
    tokenizer = None,
    per_snap_num=2000,
    max_snap_per_interval=1,
):
    # 定义长度区间
    length_intervals = {
        '0_16K': (0, 16000),
        '16K_32K': (16000, 32000),
        '32K_64K': (32000, 64000),
        '64K_128K': (64000, 128000),
    }
    length_intervals_value = {
        '0_16K': 8000, 
        '16K_32K': 24000,
        '32K_64K': 48000,
        '64K_128K': 96000,
    }
    
    # 初始化每个区间的样本计数器和数据列表
    interval_snap_counts = {interval: 0 for interval in length_intervals}
    interval_counts = {interval: 0 for interval in length_intervals}
    interval_data = {interval: [] for interval in length_intervals}
    
    cnt = 0
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    
    with tqdm(total=len(dataset)) as pbar:
        for d in dataset:
            all_docs, clue_docs = d['all_docs'], d['clue_docs']

            instruction_format = None
            if d['type'] in ['inter_doc', 'intra_doc']:
                if d['language'] == language:
                    content_key = 'Passage {pi}:\n'
                    instruction_format = 'Answer the question based on the given passages.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages and provide a complete reasoning process.\nQuestion:{q}\nAnswer:'
            else:
                if d['language'] == language:
                    content_key = 'Passage {pi}:\n'
                    instruction_format = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion:{q}\nAnswer:'
            
            pbar.set_description(f"already having {cnt} samples")
            pbar.update(1)
            
            if instruction_format is None:
                continue

            if len(clue_docs) < 2: 
                continue

            length = length_intervals_value[random.sample(length_intervals.keys(), 1)[0]]

            clue_length = calculate_clue_length(clue_docs, tokenizer)
            remain_length = length - clue_length

            concat_content = []
            total_length, id_ = 0, 0
            
            while (total_length < remain_length) and (id_ < len(all_docs)):
                total_length += len(tokenizer(all_docs[id_]['content'], add_special_tokens=False)['input_ids'])
                concat_content.append([all_docs[id_]['content'],])
                id_ += 1
            
            if (total_length < remain_length) or total_length > max_length:
                continue
            
            if len(concat_content) > len(clue_docs):
                clue_pos = random.sample(range(len(concat_content)-1), len(clue_docs))
                for p, doc in zip(clue_pos, clue_docs):
                    concat_content[p].append(doc['content'])
            else:
                continue
            
            # 计算样本的总长度
            sample_length = total_length + clue_length
            
            # 分配样本到相应的长度区间
            for interval, (min_len, max_len) in length_intervals.items():
                if min_len <= sample_length < max_len:
                    if interval_counts[interval] < per_snap_num:
                        item = {
                            "clue_pos": clue_pos,
                            "clue_docs": clue_docs,
                            "concat_content": concat_content,
                            "question": d['question'],
                            "answer": d['answer'],
                            "instruction_format": instruction_format,
                        }
                        interval_data[interval].append(item)
                        interval_counts[interval] += 1
                        cnt += 1
                    break
            
            # 检查每个区间的样本数量是否达到预设值
            for interval, data in interval_data.items():
                if len(data) >= per_snap_num:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    auto_save_data(data, os.path.join(save_dir, f'train_processed_{interval}_{len(data)}_snap{len(data)//per_snap_num}.pkl'))
                    interval_data[interval] = []  # 清空已保存的数据
                    interval_snap_counts[interval] += 1
            
            if interval_snap_counts[interval] > max_snap_per_interval:
                length_intervals.pop(interval)

            if cnt >= per_snap_num * len(length_intervals):
                break


if __name__ == "__main__":
    # model_path = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    # dataset_path = "/data/zecheng/data/LongMIT-128K"
    # save_path = "/data/zecheng/data/processed_multi_hop"

    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "donmaclean/LongMIT-128K"
    save_path = "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/data/processed_multi_hop/LongMIT-processed-v2"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    process_longmit_datasets(dataset_name=dataset_path, max_length=128000, save_dir=save_path, tokenizer=tokenizer, per_snap_num=2000, max_snap_per_interval=2)