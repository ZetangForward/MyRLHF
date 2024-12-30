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


def process_longmit_datasets(
    dataset_name: str = "/data/zecheng/data/LongMIT-128K", 
    save_dir: str = None,
    language: str = "en",
    length: int = 32000,
    max_length: int = 64000,
    tokenizer = None,
    total_num=5000,
):
    cnt = 0
    qa_pairs = []
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    with tqdm(total=len(dataset)) as pbar:
        for d in dataset:
            all_docs, clue_docs = d['all_docs'], d['clue_docs']

            instruction_format = None
            if d['type'] in ['inter_doc', 'intra_doc']:
                if d['language'] == 'en':
                    content_key = 'Passage {pi}:\n'
                    # with CoT
                    instruction_format = 'Answer the question based on the given passages.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages and provide a complete reasoning process.\nQuestion:{q}\nAnswer:'
            else:
                if d['language'] == 'en':
                    content_key = 'Passage {pi}:\n'
                    instruction_format = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\nQuestion:{q}\nAnswer:'
            pbar.set_description(f"already having {cnt} samples")
            pbar.update(1)
            if instruction_format is None:
                continue

            if len(clue_docs) < 3: 
                continue

            clue_length = calculate_clue_length(clue_docs, tokenizer)
            # clue_ids = [doc['id'] for doc in clue_docs]
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
                # concat_content = list(itertools.chain(*concat_content))
            else:
                continue
            
            item = {
                "clue_pos": clue_pos,
                "clue_docs": clue_docs,
                "concat_content": concat_content,
                "question": d['question'],
                "answer": d['answer'],
                "instruction_format": instruction_format,
            }
            qa_pairs.append(item)

            # concat_content_str = '\n'.join(concat_content)
            # question, answer = d['question'], d['answer']

            # prompt = instruction_format.format(concat_content=concat_content_str, q=question)
            # qa_pairs.append({'prompt': prompt, 'output': answer})
            cnt += 1    

            if cnt >= total_num:
                break

            if cnt % 2000 == 0:
                logger.info(f"already having {cnt} samples")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                auto_save_data(qa_pairs, os.path.join(save_dir, f'train_processed_en_snap_{cnt//2000}.pkl'))
                qa_pairs.clear()


if __name__ == "__main__":
    # model_path = "/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct"
    # dataset_path = "/data/zecheng/data/LongMIT-128K"
    # save_path = "/data/zecheng/data/processed_multi_hop"

    model_path = "/nvme/big_models/Llama-3.3-70B-Instruct"
    dataset_path = "/data/pub_data/LongMIT-128K"
    save_path = "/data/pub_data/processed_multi_hop/filter_en_for_eval"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    process_longmit_datasets(dataset_name=dataset_path, length=16000, max_length=96000, save_dir=save_path, tokenizer=tokenizer, total_num=32000)