from test_jbb import begin_test
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

# python test2_pjlab_llama_jbb.py
# nohup python test2_pjlab_llama_jbb.py > jbb.log
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_idx', type=int, default=0, help='selected index')
    parser.add_argument('--needle_path', type=str, default=None, help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='path to dataset')
    args = parser.parse_args()
    args.model_path = "/data/hf_models/Meta-Llama-3.1-8B-Instruct"
    args.adapter_path = ""#"/mnt/petrelfs/tangzecheng/local_ckpt/merge_v1/Llama-3.1-8B-Instruct/simpo/global_step325"
    args.dataset_path = "/data/pub_data/pg19-test"
    args.needle_path = "/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle_jbb_normal.jsonl"
    args.save_dir = "/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/analysis/information_flow_normal_max16k"
    args.selected_idx = [0,1]
    args.loss_type = "ce"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    needles_and_stacks = auto_read_data(args.needle_path)
    needle_list = [l["needle"] for l in needles_and_stacks]
    retrieval_question_list = [l["question"] for l in needles_and_stacks]
    evidence_list = [l["real_needle"] for l in needles_and_stacks]
    golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
    tags = [l["tag"] for l in needles_and_stacks]

    for context_length in [
        # 15900,
        11900,
          7900,
            3900, 
            1900
            ]:
        for loss_type in [
            "label",
            "ce"
                           
                           ]:
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
                noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=42)
                background_text = noise_sampler_test.get_sample(args.context_length)  # zecheng_note: 我们设置了8K上下文长度
                disturb_tok_needles = [i for i in needle if i not in evidence]
                disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))

                all_combinations = list(itertools.combinations(list(range(0, 10)), len(evidence)))
                
                logger.info(all_combinations)
                model = None
                with tqdm(total=len(all_combinations)) as pbar:
                    for depth_percent in all_combinations:
                        del model
                        torch.cuda.empty_cache()
                        # model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                        #                                              device_map='auto',
                        #                                              torch_dtype=torch.bfloat16,
                        #                                              attn_implementation="eager"
                                                # )

                        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                                     attn_implementation = "flash_attention_2"
                                                                     ).half()

                        # model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                        #                     attn_implementation="flash_attention_2",
                        #                     # device_map = "auto",
                        #                     ).half().cuda().eval()
                        layer_name = model.model.layers[0].__class__.__name__
                        max_memory = get_balanced_memory(
                            model,
                            max_memory=None,
                            no_split_module_classes=[layer_name],
                            dtype='float16',
                            low_zero=False,
                        )

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
                        print(model)
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
                        
                        save_file_name = f"{model_name}/{args.context_length}/{args.loss_type}/{tag}_{depth_tag}"
                        
                        begin_test(args, question, answer, s_id, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, with_adapter= True if args.adapter_path else False,
                                   start_layer = 24)
                        pbar.update(1)
                        # exit(0)

        print("OVER:",context_length, loss_type)