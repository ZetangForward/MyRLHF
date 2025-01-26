from test_jbb_retain_gpu015 import find_multi_needle_idx, get_random_emoji, random_combine
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from modelzipper.tutils import *
import itertools
from peft import peft_model, PeftModelForCausalLM
import numpy as np
import datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(sys.path)
from retrieval_head_detection import SentenceSampler
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# from ..utils import get_random_emoji
# python test2_pjlab_llama_jbb.py
# nohup python test2_pjlab_llama_jbb_random5x100.py > jbb.log

# nohup python test2_pjlab_llama_jbb_random5x100_find_span.py > jbb_retain.log 2>&1 &



#  python test2_pjlab_llama_jbb_random5x100_find_span.py
def combine_pkl_files_to_one(dir_name, delete = True):
    ret = {}
    for file_name in os.listdir(dir_name):
        if file_name =="full.pkl":
            continue
        file_path = os.path.join(dir_name, file_name)
        ret[file_name] = pickle.load(open(file_path,"rb"))

        if delete:
            os.remove(file_path)

    pickle.dump(ret, open(f"{dir_name}/full.pkl","wb"))


def extract_pkl_files_from_one(dir_name, delete = True):
    path = f"{dir_name}/full.pkl"
    if not os.path.exists(path):return
    files = pickle.load(open(path,"rb"))

    for file_name, file in files.items():
        if file_name =="full.pkl":
            continue
        file_path = os.path.join(dir_name, file_name)
        pickle.dump(file, open(file_path,"wb"))
    
    if delete:
        os.remove(f"{dir_name}/full.pkl")

def begin_test(args, question, answer, selected_idx, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, is_0k, use_emoji,
showlog=True):
    if background_text is not None:

        if use_emoji:
            emojis10 = get_random_emoji(tokenizer, 10, return_idx = True)
            background_text, emoji_pos = random_combine(background_text, emojis10, 
                                                         return_snd_pos = True)
            emoji_pos = set(emoji_pos)
            cumsum_num = 0
            emoji_spans = []


        depth_percent = [i / 10 for i in depth_percent]
        updated_sample = [[] for _ in range(len(background_text) + 1)]
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(evidence, real_pos):  # insert real needle
            updated_sample[pos].append(fact)
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)


    
        for i, s in enumerate(background_text):  # insert irrevelent needle
            if use_emoji and (i in emoji_pos):

                cur_pos = sum((len(l) for l in updated_sample[i]), 0)
                emoji_spans +=[(cumsum_num + cur_pos, cumsum_num + cur_pos + len(s))]

            updated_sample[i].append(s)

            if use_emoji:
                cumsum_num += sum((len(l) for l in updated_sample[i]), 0)
    else:
        updated_sample = random_combine(evidence[:-1], disturb_tok_needles+[evidence[-1]])
        updated_sample = [[k] for k in updated_sample]
        # print("updated_sample:", updated_sample)
    
    if not use_emoji or is_0k:
        emoji_spans = []
        

    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\n{question}\nAnswer:"
    if tokenizer.chat_template is not None:
        shift = 30
        inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
    else:
        shift = 0
        inp = tokenizer(input_context, return_tensors='pt').input_ids
    emoji_spans = [(k[0] + shift, k[1] + shift) for k in emoji_spans]
    
    # if use_emoji:
    #     print("emoji:")

    #     for emoji_span, emj in zip(emoji_spans,emojis10):
    #         print("O:",tokenizer.decode(emj),emj)
    #         print("N:",tokenizer.decode(inp[0,emoji_span[0]:emoji_span[1]].tolist()),inp[0,emoji_span[0]:emoji_span[1]].tolist())
    #         print()

    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_list[selected_idx],showlog=showlog)
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles,showlog=showlog)

    return search_pos, attack_pos, emoji_spans
    inp = inp.to(model.device)
    
    with torch.no_grad():
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    if tokenizer.chat_template is not None:
        inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_context}, {"role": "assistant", "content": answer}], 
            tokenize=True, add_generation_prompt=False, return_tensors='pt'
        ).to(model.device)
    else:
        inp = tokenizer(input_context + "\n" + answer, return_tensors='pt').input_ids.to(model.device)
    

    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    toks_length = answer_ids.size(-1)
    for j in range(inp.size(-1), toks_length, -1):
        if (inp[0, j-toks_length : j] == answer_ids).sum().item() == toks_length:
            target_pos = (j-toks_length, j) 
            break
    else:
        raise ValueError("Not find target in input tokens!")
    # print("ANDADSDSAD:", inp.shape, target_pos, answer_ids, toks_length,answer)
    


#   python test2_pjlab_llama_jbb_random5x100_find_span.py
if __name__ == "__main__":
    print("Pid:",os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_idx', type=int, default=0, help='selected index')
    parser.add_argument('--needle_path', type=str, default=None, help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='path to dataset')
    args = parser.parse_args()
    # args.model_path = "/data/hf_models/meta-llama-3.1-8b"
    args.model_path = "/data/hf_models/meta-llama-3.1-8b"

    model_save_name = "meta-llama-3.1-8b"
    args.adapter_path = ""#"/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step325"
    args.dataset_path = "/data/pub_data/pg19-test"
    args.needle_path = "/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle_new.jsonl"
    # args.save_dir = "/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/analysis/information_flow_normal_max12k_sample200_gws"
    # args.save_dir ="/data/zecheng/acl2025/Long-form-reasoning/preliminary/babilong_random5x100/results/information_flow_normal_max12k_sample200_gws"
    args.save_dir ="/data/zecheng/acl2025/Long-form-reasoning/preliminary/babilong_random5x100/results/information_flow_normal_max12k_sample200_gws/"
    args.selected_idx = list(range(0, 100))

    args.use_emoji = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    needles_and_stacks = auto_read_data(args.needle_path)
    needle_list = [l["needle"] for l in needles_and_stacks]
    retrieval_question_list = [l["question"] for l in needles_and_stacks]
    evidence_list = [l["real_needle"] for l in needles_and_stacks]
    golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
    tags = [l["tag"] for l in needles_and_stacks]


    for pe,pn in zip(evidence_list, needle_list):
        last_idx = pn.index(pe[-1])
        assert last_idx > -1

        pe += [pn[last_idx + 1]]
    
        # print("evidence:", pe)

    root_dir = args.save_dir + "/" + model_save_name
    all_dic = {}
    random.seed(42)
    for context_length in os.listdir(root_dir):
        # print(context_length)
        # continue
        dic = {}
        for loss_type in [ "label" ]:
            args.context_length = int(context_length)
            args.loss_type = loss_type
            path = os.path.join(root_dir,context_length,args.loss_type)

            for file_name in tqdm(os.listdir(path)):
                s_id = int(re.search(r"hop_sid-(\d*)_",file_name).group(1))

                # logger.info(f"Selected idx: {s_id}")
                # logger.info(f"Question: {retrieval_question_list[s_id]}")
                # logger.info(f"Answer: {golden_answer_list[s_id]}")
                # logger.info(f"Tag: {tags[s_id]}")
                # logger.info(f"Needle: {needle_list[s_id]}")
                # logger.info(f"Real Needle: {evidence_list[s_id]}")
                # logger.info("=============================================")



                needle = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in needle_list[s_id]]
                evidence = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in evidence_list[s_id]]
                question = retrieval_question_list[s_id]
                answer = golden_answer_list[s_id]
                tag = tags[s_id]

                # 初始化采样器
                haystack = datasets.load_dataset(args.dataset_path, split="test")
                if args.context_length>0:
                    noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=42)
                    background_text = noise_sampler_test.get_sample(args.context_length)  
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    np.random.seed(42)
                    disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))
                else:
                    background_text = None
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    disturb_pos = None


                model_name = args.model_path.split("/")[-1]

                depth_percent = tuple(map(int,re.search("pid-(\d*)_(.*).pkl", file_name).group(2).split("-")))

                

                
                search_pos, attack_pos, emoji_spans = begin_test(args, question, answer, s_id, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, file_name, model_name, is_0k = (args.context_length == 0), use_emoji = args.use_emoji,
                                                                 showlog=False)
                
                dic[file_name]={
                    "reference":search_pos,
                    "attack": attack_pos,
                    "emoji": emoji_spans,
                    "ctx_length": 0 if args.context_length == 0 else (args.context_length + 100)
                }
                # print(search_pos,attack_pos,emoji_spans)
        
        all_dic[context_length] = dic

        print("OVER:",context_length, loss_type)

    json.dump(all_dic, open(f"{args.save_dir}/{model_save_name}/length.json","w"))