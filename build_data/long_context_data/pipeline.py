import os
from modelzipper.tutils import *
import datasets
import torch
import numpy as np
import transformers
import pickle
def merge_intervals(intervals):
    if intervals.size(0) == 0:
        return intervals

    start = intervals[:, 0]
    end = intervals[:, 1]
    adjacent = (start[1:] - end[:-1]) == 0

    keep_start_mask = torch.cat([torch.tensor([True]), ~adjacent])
    merged_start = start[keep_start_mask]
    keep_end_mask = torch.cat([~adjacent, torch.tensor([True])])
    merged_end = end[keep_end_mask]

    merged_intervals = torch.stack([merged_start, merged_end], dim=1)
    
    return merged_intervals 

@torch.no_grad
def find_key_token(text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path=None, question_pos=None, answer_pos=None):
    text_encoded = evaluator_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = text_encoded['input_ids'].to(evaluator_model.device)
    
    output_full = evaluator_model(input_ids)
    
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    _, max_len = input_ids.shape
    key_tokens = []

    chunk_num = int(np.ceil(max_len / sliding_window))
    with tqdm(total=chunk_num) as pbar:
        for i, start_token in enumerate(range(0, max_len-trunc_len, sliding_window)):
            if start_token+trunc_len+sliding_window > max_len:
                sliding_window = max_len-start_token-trunc_len
            input_ids_short = input_ids[:, start_token: start_token+trunc_len+sliding_window]
            output_short = evaluator_model(input_ids_short)

            loss_full = loss_f(output_full.logits[0, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :], input_ids[0, start_token+trunc_len: start_token+trunc_len+sliding_window])
            loss_short = loss_f(output_short.logits[0, trunc_len-1: trunc_len+sliding_window-1, :], input_ids_short[0, trunc_len: trunc_len+sliding_window])

            loss_discrepancy = (torch.logical_and((loss_short - loss_full) > 2, loss_full < 2)).squeeze()

            for i, is_key in enumerate(loss_discrepancy):
                if is_key:
                    key_tokens.append(start_token+trunc_len+i)

            pbar.update(1)

    key_text_intervals = merge_intervals(text_encoded['offset_mapping'][0, key_tokens])

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            slices_str = ";".join([f"[{element[0]}, {element[1]}]" for element in key_text_intervals])
            f.write(slices_str)
    return key_text_intervals

def load_key_token(save_path):
    with open(save_path, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            key_slices_str = line.split(';')
            key_text_slices = []
            for key_slice in key_slices_str:
                key_text_slices.append(eval(key_slice))
            return key_text_slices

def cal_overlap(offset_mapping, key_text_slices):
    if key_text_slices is None:
        return None

    key_tokens = []
    i, j = 0, 0
    
    while i < len(offset_mapping) and j < len(key_text_slices):
        a_start, a_end = offset_mapping[i]
        b_start, b_end = key_text_slices[j]

        if a_start >= b_start and a_end <= b_end:
            key_tokens.append(i-1)
            i += 1
        elif a_start < b_start:
            i += 1
        else:
            j += 1

    return key_tokens


def search_token_id(st, ed, offset_mapping):
    r"""
    return the token id of the first token that overlaps with the given range
    
    Parameters:
        st (`int`): start offset
        ed (`int`): end offset
        offset_mapping (`torch.Tensor`): offset mapping of the text
    """
    i = 0
    token_ids = []
    while i < len(offset_mapping):
        cur_st, cur_ed = offset_mapping[i]
        if st >= cur_st and cur_ed <= ed:
            token_ids.append(i)
    return token_ids[0], token_ids[-1]


def locate_question_answer_offset(s_c, s_q, s_a, offset_mapping):
    r"""
    locate question and answer offset in the context
    
    Args:
        s_c (`str`): context
        s_q (`str`): question
        s_a (`str`): answer
        offset_mapping (`torch.Tensor`): offset mapping
    """
    st_q, st_a = s_c.index(s_q), s_c.index(s_a)
    ed_q, ed_a = st_q + len(s_q), st_a + len(s_a)
    question_pos = search_token_id(st_q, ed_q, offset_mapping)
    answer_pos = search_token_id(st_a, ed_a, offset_mapping)
    return question_pos, answer_pos


def preprocess_item(item, tokenizer, device):
    question = item[0]['content'].split('\r\n\n')[1]
    answer = item[1]['content']

    input_text = tokenizer.apply_chat_template(item[:2], add_generation_prompt=False, tokenize=False)
    encoded_input = tokenizer(input_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)

    input_ids = encoded_input['input_ids'].to(device)
    offset_mapping = encoded_input['offset_mapping'][0]

    question_pos, answer_pos = locate_question_answer_offset(input_text, question, answer, offset_mapping)

    return input_text, input_ids, question_pos, answer_pos, offset_mapping

@torch.no_grad
def compute_longppl(
        item,
        model,
        evaluator_model,
        tokenizer=None, 
        evaluator_tokenizer=None, 
        save_path=None, 
        trunc_len=4096, 
        sliding_window=1024
    ):
    r"""
    Compute the LongPPL for long text sequences.

    Parameters:
        text (`str`): 
            The input text for which LongPPL is calculated.
        model (`transformers.PretrainedModel` or `str`): 
            Can be either:
                - The model used for LongPPL calculation.
                - The path to the model used for LongPPL calculation.
        evaluator_model (`transformers.PretrainedModel` or `str`): 
            Can be either:
                - The evaluator model used to identify the key tokens.
                - The path to the evaluator model.
        tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
            Tokenizer of the evaluated model if `model` is specified with a `transformers.PretrainedModel` object, otherwise should be `None`.
        evaluator_tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
            Tokenizer of the evaluator model if `evaluator_model` is specified with a `transformers.PretrainedModel` object, otherwise should be `None`.
        save_path (`str`, *optional*): If specified, the path to save the computed key tokens.
        trunc_len (`int`, *optional*, default=4096): Length of the truncated short context.
        sliding_window (`int`, *optional*, default=1024): Number of tokens sharing the same short context.

    Returns:
        [`Dict[np.float32, int, np.float32, int]`]: A `Dict` object including:
            - 'longppl' (`np.float32`): The LongPPL score.
            - 'n_key_token' (`int`): The number of key tokens (under the evaluated model).
            - 'ppl' (`np.float32`): The PPL score.
            - 'n_token' (`int`): The number of tokens in the input text.
    """
    text, input_ids, question_pos, answer_pos, offset_mapping = preprocess_item(item, tokenizer, model.device)
    logger.info(f'Input length: {input_ids.shape}')
    
    if evaluator_model is not None:
        key_text_slices = find_key_token(input_ids, offset_mapping, evaluator_model, trunc_len, sliding_window, save_path, question_pos, answer_pos)
    else:
        key_text_slices = load_key_token(save_path)

    key_tokens = cal_overlap(offset_mapping, key_text_slices)
    str_key_tokens = [tokenizer.decode(token_id) for token_id in key_tokens]
    logger.info(str_key_tokens)
    
    output_full = model(input_ids)

    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
    
    if key_tokens is None or len(key_tokens) == 0:
        print("No key tokens!")
        return {"longppl": None, "n_key_token": None, "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}

    loss_key = loss_overall[key_tokens]

    return {"longppl": np.exp(loss_key.mean()), "n_key_token": len(key_tokens), "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}


if __name__ == '__main__':
    dir_path = '/mnt/hwfile/opendatalab/tangzecheng/long-context-gpt-build-data/gpt'
    all_files = auto_read_dir(dir_path, file_suffix='json')
    # content = datasets.load_dataset('json', data_files=os.path.join(dir_path, all_files[0]), split='train')['conversations']
    
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:0')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    logger.info('begin to load datasets')
    with open("/mnt/petrelfs/tangzecheng/MyRLHF/build_data/long_context_data/test_sample.pkl", "rb") as f:
        test_case = pickle.load(f)
    # test_case = content[0][0]['content']  # DEBUG
    # import pdb; pdb.set_trace()
    res = compute_longppl(test_case, model, model, tokenizer, tokenizer)
    print(res)