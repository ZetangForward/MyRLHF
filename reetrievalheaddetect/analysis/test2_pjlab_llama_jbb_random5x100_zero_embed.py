from test_jbb_retain_zero_embed import begin_test
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

# nohup python test2_pjlab_llama_jbb_random5x100_zero_embed.py > jbb_zero_embed.log 2>&1 &

#  nohup python test2_pjlab_llama_jbb_random5x100_retain_grad_gpu015.py > jbb_retain.log && python test2_pjlab_llama_jbb_random5x100_zero_embed.py > jbb_zero_embed.log
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


# nohup python analysis/test2_pjlab_llama_jbb_random5x100_zero_embed.py --model_path="meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_path="/mnt/petrelfs/tangzecheng/local_data/pg19-test" --save_dir="/mnt/petrelfs/tangzecheng/repos/SaliencyResults/preliminary/babilong_random5x100/results/information_flow_normal_max12k_sample100_gws" --testing_lengths 3900 7900 > ./logs/embedding_zero.log 2>&1 &

if __name__ == "__main__":
    print("Pid:",os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_idx', type=int, default=0, help='selected index')
    parser.add_argument('--needle_path', type=str, default=None, help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='path to dataset')
    parser.add_argument('--testing_lengths', type=int, nargs='+', default=None, help="A list of integers")
    args = parser.parse_args()
    # args.model_path = "/data/hf_models/meta-llama-3.1-8b"
    
    # args.model_path = "/data/hf_models/Meta-Llama-3.1-8B-Instruct"
    args.factor = 0.01
    args.adapter_path = ""#"/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step325"
    # args.dataset_path = "/data/pub_data/pg19-test"
    args.needle_path = "./haystack_for_detect/reasoning_needle_new.jsonl"
    # args.save_dir ="/data/zecheng/acl2025/Long-form-reasoning/preliminary/babilong_random5x100/results/information_flow_normal_max12k_sample200_gws"
    
    args.save_dir ="/data/zecheng/acl2025/Long-form-reasoning/preliminary/babilong_random5x100/results/information_flow_normal_max12k_sample3x100_gws_control"
    args.selected_idx = list(range(1, 200, 2))

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

    random.seed(42)
    for context_length in [
        11900,
        # 7900,
        # 3900,
        # 1900,
            # 900,
        
            ]:
        for loss_type in [ "label" ]:
            args.context_length = context_length
            args.loss_type = loss_type
            for s_id in args.selected_idx:
                logger.info(f"Selected idx: {s_id}")
                logger.info(f"Question: {retrieval_question_list[s_id]}")
                logger.info(f"Answer: {golden_answer_list[s_id]}")
                logger.info(f"Tag: {tags[s_id]}")
                logger.info(f"Needle: {needle_list[s_id]}")
                logger.info(f"Real Needle: {evidence_list[s_id]}")
                logger.info("=============================================")

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
                    print("disturb:",disturb_pos)
                else:
                    background_text = None
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    disturb_pos = None
                # combinations_number = 5
                # all_combinations = list(itertools.combinations(list(range(10)), len(evidence)))
                # all_combinations = random.sample(all_combinations, combinations_number)

                combinations_number = 100
                all_combinations = list(itertools.combinations(list(range(10)), len(evidence)))
                all_combinations = random.sample(all_combinations, combinations_number)
                # all_combinations = [(0, 2, 4, 6, 8), (1, 2, 3, 4, 5),(5, 6, 7, 8, 9)]
                model = None
                cnt = 0
                with tqdm(total=len(all_combinations)) as pbar:
                    for _, depth_percent in enumerate(all_combinations):
                        # random.shuffle(combinations)
                        # depth_percent = depth_percent[:len(evidence)]

                        del model
                        torch.cuda.empty_cache()
                        if cnt == 3: break
                        # model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                        #                                              device_map='auto',
                        #                                              torch_dtype=torch.bfloat16,
                        #                                              attn_implementation="eager"
                                                # )
                        try:
                            model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                                        attn_implementation = "flash_attention_2"
                                                                        ).half()

                            # model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                            #                     attn_implementation="flash_attention_2",
                            #                     # device_map = "auto",
                            #                     ).half().cuda().eval()
                            # layer_name = model.model.layers[0].__class__.__name__
                            # max_memory = get_balanced_memory(
                            #     model,
                            #     max_memory=None,
                            #     no_split_module_classes=[layer_name],
                            #     dtype='float16',
                            #     low_zero=False,
                            # )

                            device_map = {
                                "model.embed_tokens": 0,
                                "model.rotary_emb" :0,

                                "model.layers.0" :0,
                                "model.layers.1" :0,
                                "model.layers.2" :0,
                                "model.layers.3" :1,

                                "model.layers.4" :1,
                                "model.layers.5" :1,
                                "model.layers.6" :2,
                                "model.layers.7" :2,

                                "model.layers.8" :2,
                                "model.layers.9" :3,
                                "model.layers.10" :3,
                                "model.layers.11" :3,

                                "model.layers.12" :4,
                                "model.layers.13" :4,
                                "model.layers.14" :4,
                                "model.layers.15" :5,

                                "model.layers.16" :5,
                                "model.layers.17" :5,
                                "model.layers.18" :6,
                                "model.layers.19" :6,

                                "model.layers.20" :6,
                                "model.layers.21" :7,
                                "model.layers.22" :7,
                                "model.layers.23" :7,
                                
                                "model.layers.24" :0,
                                "model.layers.25" :1,
                                "model.layers.26" :2,
                                "model.layers.27" :3,
                                
                                "model.layers.28" :4,
                                "model.layers.29" :5,
                                "model.layers.30" :6,
                                "model.layers.31" :7,
                                "model.norm" :6,
                                "lm_head"  : 7
                                
                            }
                            # print(model)
                            # device_map = infer_auto_device_map(
                            #     model,
                            #     max_memory=max_memory,
                            #     no_split_module_classes=[layer_name],
                            #     dtype='float16'
                            # )

                            model = dispatch_model(model, device_map=device_map)

                            # for param in model.parameters():
                            #     param.requires_grad = False

                            pbar.set_description(f"Processing depth {depth_percent}")
                            depth_tag = "-".join([str(i) for i in depth_percent])
                            model_name = args.model_path.split("/")[-1]
                            
                            save_file_name = f"{model_name}_factor{args.factor}/{args.context_length}/{args.loss_type}/{tag}_sid-{s_id}_pid-{cnt}_{depth_tag}"
                            
                            begin_test(args, question, answer, s_id, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, is_0k = (context_length == 0), use_emoji = args.use_emoji, with_adapter= True if args.adapter_path else False,                                   start_layer = 24,
                                    factor = args.factor)
                            pbar.update(1)
                            cnt += 1
                            
                            print("dep_p:",depth_percent)
                        except ZeroDivisionError as ze:
                            continue

                        except ValueError as e:
                            if str(e) =="evidence_list and disturb_tok_needles length not match!":
                                continue
                    if cnt != 5:
                        print(f"args.context_length: {args.context_length}")
                        print(f"args.loss_type: {args.loss_type}")
                        print(f"cnt: {cnt}")
                        print(f"s_id: {s_id}")

            
            print("Start merge:")
            file_dir =f"{args.save_dir}/{model_name}_factor{args.factor}/{args.context_length}/{args.loss_type}/"
            # combine_pkl_files_to_one(file_dir)

        print("OVER:",context_length, loss_type)

    
