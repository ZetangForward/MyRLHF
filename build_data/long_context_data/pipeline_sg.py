import os
from modelzipper.tutils import *
import datasets
import torch
import copy
import numpy as np
import transformers
import matplotlib.pyplot as plt

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
def find_key_token(input_ids, offset_mapping, model, trunc_len, sliding_window, question_pos=None, answer_pos=None, theta=2.0, save_path=None):
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    output_full = model(input_ids)
    loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
    ppl_full = np.exp(loss_overall.mean())

    _, max_len = input_ids.shape
    key_tokens = []

    chunk_num = int(np.ceil((max_len - trunc_len)) / sliding_window)
    question_ipt_ids = input_ids[:, question_pos[0]: question_pos[1]]
    answer_ipt_ids = input_ids[:, answer_pos[0]: answer_pos[1]]

    with tqdm(total=chunk_num) as pbar:
        for i, start_token in enumerate(range(0, max_len-trunc_len, sliding_window)):
            if start_token+trunc_len+sliding_window > max_len:
                sliding_window = max_len-start_token-trunc_len
            # create short input
            sliding_ipt_ids = input_ids[:, start_token: start_token+trunc_len+sliding_window]
            combine_short_ipt_ids = torch.cat([sliding_ipt_ids, question_ipt_ids, answer_ipt_ids], dim=1)
            # compute short logits PPL
            output_short = model(combine_short_ipt_ids)
            loss_short = loss_f(
                output_short.logits[0, trunc_len-1: trunc_len+sliding_window-1, :], 
                combine_short_ipt_ids[0, trunc_len: trunc_len+sliding_window]
            )

            loss_full = loss_f(
                output_full.logits[0, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :], 
                input_ids[0, start_token+trunc_len: start_token+trunc_len+sliding_window]
            )

            tmp = loss_short - loss_full
            plot_distribution(tmp, os.path.join("/mnt/petrelfs/tangzecheng/MyRLHF/build_data/long_context_data/case_figs", f"chunk_{i}.png"))
            loss_discrepancy = (torch.logical_and((loss_short - loss_full) > theta, loss_full < theta)).squeeze()

            for i, is_key in enumerate(loss_discrepancy):
                if is_key:
                    key_tokens.append(start_token+trunc_len+i)

            pbar.update(1)
    key_text_intervals = merge_intervals(offset_mapping[key_tokens])

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            slices_str = ";".join([f"[{element[0]}, {element[1]}]" for element in key_text_intervals])
            f.write(slices_str)
    return key_text_intervals, ppl_full, loss_overall

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


def search_token_id(st, ed, offset_mapping, s_c=None):
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
        cur_st, cur_ed = offset_mapping[i][0], offset_mapping[i][1]
        
        if (cur_st <= st and cur_ed >= st) or (cur_st <= ed and cur_ed >= ed) or (cur_st >= st and cur_ed <= ed):
            token_ids.append(i)
        if cur_ed >= ed:
            token_ids.append(i)
            break
        i += 1
    
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
    question_pos = search_token_id(st_q, ed_q, offset_mapping, s_c)
    answer_pos = search_token_id(st_a, ed_a, offset_mapping, s_c)
    return question_pos, answer_pos


@torch.no_grad
def model_prediction(message, model, tokenizer, device):
    input_ids = tokenizer(message, return_tensors="pt").input_ids.to(device)
    model_pred = model.generate(input_ids, max_new_tokens=100, do_sample=True)[0]
    return tokenizer.decode(model_pred[input_ids.size(1):], skip_special_tokens=True)


def preprocess_item(item, model, tokenizer, device):
    question = item[0]['content'].split('\r\n\n')[1]
    answer = item[1]['content']

    logger.info(f"question: {question}")
    logger.info(f"answer: {answer}")
    
    import pdb; pdb.set_trace()
    golden_input_text = tokenizer.apply_chat_template(item[:2], add_generation_prompt=False, tokenize=False)
    input_query = tokenizer.apply_chat_template(item[:1], add_generation_prompt=True, tokenize=False)
    pred_str = model_prediction(input_query, model, tokenizer, device)
    logger.info(f"model prediction: {pred_str}")
    
    model_pred_message = copy.deepcopy(item[:2])
    model_pred_message[1]['content'] = pred_str
    model_pred_text = tokenizer.apply_chat_template(model_pred_message, add_generation_prompt=False, tokenize=False)

    encoded_golden_input = tokenizer(golden_input_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    encoded_pred_input = tokenizer(model_pred_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)

    golden_input_ids = encoded_golden_input['input_ids'].to(device)
    pred_input_ids = encoded_pred_input['input_ids'].to(device)
    golden_offset_mapping = encoded_golden_input['offset_mapping'][0]
    pred_offset_mapping = encoded_pred_input['offset_mapping'][0]

    golden_question_pos, golden_answer_pos = locate_question_answer_offset(golden_input_text, question, answer, golden_offset_mapping)
    pred_question_pos, pred_answer_pos = locate_question_answer_offset(model_pred_text, question, pred_str, pred_offset_mapping)
   
    return (
        golden_input_text, model_pred_text,
        golden_input_ids, pred_input_ids,
        golden_question_pos, pred_question_pos,
        golden_answer_pos, pred_answer_pos,
        golden_offset_mapping, pred_offset_mapping
    )


def plot_distribution(data_tensor, save_path):
    """
    绘制数据的分布图。
    
    参数:
        data_tensor (torch.Tensor): 要绘制的数据，应为一维。
    """
    # 将tensor转换为numpy array，因为matplotlib更容易与numpy配合
    data_array = data_tensor.float().cpu().numpy()

    # 创建一个直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data_array, bins=50, alpha=0.75)
    plt.title('Distribution of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)


@torch.no_grad
def compute_longppl(
        item,
        model,
        tokenizer=None,
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
    # text, input_ids, question_pos, answer_pos, offset_mapping = preprocess_item(item, model, tokenizer, model.device)

    golden_input_text, model_pred_text, golden_input_ids, pred_input_ids, golden_question_pos, pred_question_pos, golden_answer_pos, pred_answer_pos, golden_offset_mapping, pred_offset_mapping = preprocess_item(item, model, tokenizer, model.device) 
    
    torch.cuda.empty_cache()
    
    if model is not None:
        logger.info("search key tokens within the golden answer")
        golden_key_text_slices, golden_ppl_full, golden_loss_overall = find_key_token(golden_input_ids, golden_offset_mapping, model, trunc_len, sliding_window, golden_question_pos, golden_answer_pos, 2.0, save_path)
        torch.cuda.empty_cache()
        logger.info("search key tokens within the model prediction")
        pred_key_text_slices, pred_ppl_full, pred_loss_overall = find_key_token(pred_input_ids, pred_offset_mapping, model, trunc_len, sliding_window, pred_question_pos, pred_answer_pos, 2.0, save_path)
    else:
        key_text_slices = load_key_token(save_path)
    
    golden_key_tokens = cal_overlap(golden_offset_mapping, golden_key_text_slices)
    pred_key_tokens = cal_overlap(pred_offset_mapping, pred_key_text_slices)

    golden_str_key_tokens = [tokenizer.decode(token_id) for token_id in golden_key_tokens]
    logger.info(golden_str_key_tokens)

    pred_str_key_tokens = [tokenizer.decode(token_id) for token_id in pred_key_tokens]
    logger.info(pred_str_key_tokens)
    
    
    if golden_key_tokens is None or len(golden_str_key_tokens) == 0:
        logger.info("No essential tokens within the context with golden prediction")
        return {"longppl": None, "n_key_token": None, "ppl": golden_ppl_full, "n_token": golden_input_ids.shape[-1]}

    if pred_key_tokens is None or len(pred_str_key_tokens) == 0:
        logger.info("No essential tokens within the context with self-generated text")
        return {"longppl": None, "n_key_token": None, "ppl": pred_ppl_full, "n_token": pred_input_ids.shape[-1]}

    golden_loss_key, pred_loss_key = golden_loss_overall[golden_key_tokens], pred_loss_overall[pred_key_tokens]

    return {
        "golden_longppl": np.exp(golden_loss_key.mean()), 
        "pred_longppl": np.exp(pred_loss_key.mean()), 
        "n_key_token (w golden answer)": len(golden_key_tokens), 
        "n_key_token (w model self-prediction)": len(pred_key_tokens),
        "ppl (w golden answer)": np.exp(golden_loss_overall.mean()), 
        "ppl (w model self-prediction)": np.exp(pred_loss_overall.mean()), 
        "n_token (w golden answer)": golden_input_ids.shape[-1],
        "n_token (w model self-prediction)": pred_input_ids.shape[-1],
    }


if __name__ == '__main__':
    dir_path = '/mnt/hwfile/opendatalab/tangzecheng/long-context-gpt-build-data/gpt'
    all_files = auto_read_dir(dir_path, file_suffix='json')
    # content = datasets.load_dataset('json', data_files=os.path.join(dir_path, all_files[0]), split='train')['conversations']
    
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda:0')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    logger.info('begin to load datasets')
    with open("/mnt/petrelfs/tangzecheng/MyRLHF/build_data/long_context_data/test_sample.pkl", "rb") as f:
        test_case = pickle.load(f)
    # test_case = content[0][0]['content']  # DEBUG
    
    res = compute_longppl(test_case, model, tokenizer)
    print(res)