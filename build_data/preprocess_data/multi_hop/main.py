from datasets import load_dataset
import json
from multiprocessing import Pool, cpu_count
import os
from transformers import AutoTokenizer
from modelzipper.tutils import *
import itertools


def calculate_clue_length(clue_docs, tokenizer):
    total_length = 0
    for doc in clue_docs:
        total_length += len(tokenizer(doc['content'], add_special_tokens=False)['input_ids'])
    return total_length


def process_single_item(args):
    d, tokenizer, length = args
    all_docs, clue_docs = d['all_docs'], d['clue_docs']
    if len(clue_docs) < 3:
        return None

    clue_length = calculate_clue_length(clue_docs, tokenizer)
    remain_length = length - clue_length

    concat_content = []
    total_length, id_ = 0, 0

    while (total_length < remain_length) and (id_ < len(all_docs)):
        total_length += len(tokenizer(all_docs[id_]['content'], add_special_tokens=False)['input_ids'])
        concat_content.append([all_docs[id_]['content']])
        id_ += 1

    if total_length < remain_length or len(concat_content) <= len(clue_docs):
        return None

    clue_pos = random.sample(range(len(concat_content) - 1), len(clue_docs))
    for p, doc in zip(clue_pos, clue_docs):
        concat_content[p].append(doc['content'])

    instruction_format = (
        'Answer the question based on the given passages.\n\nThe following are given passages.\n{concat_content}\n\n'
        'Answer the question based on the given passages and provide a complete reasoning process.\nQuestion:{q}\nAnswer:'
        if d['type'] in ['inter_doc', 'intra_doc'] and d['language'] == 'en'
        else None
    )

    if instruction_format is None:
        return None

    return {
        "clue_pos": clue_pos,
        "clue_docs": clue_docs,
        "concat_content": concat_content,
        "question": d['question'],
        "answer": d['answer'],
        "instruction_format": instruction_format,
    }


def process_longmit_datasets_multiprocess(
    dataset_name: str = "/data/zecheng/data/LongMIT-128K",
    save_dir: str = None,
    language: str = "en",
    length: int = 64000,
    tokenizer=None,
    total_num=5000,
    num_workers=None,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    args = [(d, tokenizer, length) for d in dataset]
    qa_pairs = []
    cnt = 0

    with Pool(num_workers) as pool:
        with tqdm(total=len(dataset)) as pbar:
            for result in pool.imap(process_single_item, args):
                pbar.update(1)
                if result is None:
                    continue

                qa_pairs.append(result)
                cnt += 1

                if cnt >= total_num:
                    break

                if cnt % 1000 == 0:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    auto_save_data(qa_pairs, os.path.join(save_dir, f'train_processed_en_snap_{cnt//1000}.pkl'))
                    qa_pairs.clear()

    # if qa_pairs and save_dir:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     auto_save_data(qa_pairs, os.path.join(save_dir, 'train_processed_en_final.pkl'))

            

if __name__ == "__main__":
    model_path = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/data/zecheng/data/LongMIT-128K"
    save_path = "/data/zecheng/data/processed_multi_hop"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    process_longmit_datasets_multiprocess(dataset_name=dataset_path, save_dir=save_path, tokenizer=tokenizer, total_num=8000, num_workers=64)