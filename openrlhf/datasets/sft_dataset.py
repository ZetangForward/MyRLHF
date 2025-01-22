from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils import zero_pad_sequences


def preprocess_data(data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, meta_key=None):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
            if meta_key: # FIXME: 这里应该数据集传入的时候就有对应的question和answer，但是我没有处理好，后面的数据要自带question和clues，不用再手动处理了
                question = data[input_key][0]['content'].split("\n")[-2]
                content_key = 'Passage {pi}:\n'
                # with CoT
                instruction_format = 'Answer the question based on the given passages.\n\nThe following are given passages.\n{concat_content}\n\nAnswer the question based on the given passages and provide a complete reasoning process.\nQuestion:{q}\nAnswer:'
                concat_content = '\n'.join([content_key.format(pi=di+1)+doc for di, doc in enumerate(data[meta_key])])
                clue_prompt = instruction_format.format(concat_content=concat_content, q=question)
                clue_prompt = apply_chat_template([{"role": "user", "content": clue_prompt}], tokenize=False)
    else:
        prompt, response = data[input_key][0]["content"], data[input_key][1]["content"]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        # response = data[output_key] if output_key else ""

    if meta_key:
        return prompt, response, clue_prompt, data[meta_key]
    return prompt, response, None, None


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiple_of=1,
        search_clue_seg=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.search_clue_seg = search_clue_seg

        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors, 
            load_from_cache_file=False, cache_file_name=None  # 禁用缓存
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None, cache_file_name=None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        if self.search_clue_seg:
            self.clue_prompts = processed_dataset["clue_prompt"]
            self.clue_prompt_ids_lens = processed_dataset["clue_prompt_ids_len"]
            self.clue_poss = processed_dataset["clue_pos"]
        else:
            self.clue_prompts = None
            self.clue_prompt_ids_lens = None
            self.clue_poss = None

    def process_data(self, data):
        if self.search_clue_seg:
            meta_key = "meta_info"  # zecheng note： 这里先这么规定
        else:
            meta_key = None
        prompt, response, clue_prompt, clue_list = preprocess_data(
            data,
            self.input_template,
            self.input_key,
            self.output_key,
            self.apply_chat_template,
            meta_key=meta_key,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        else:
            prompt_ids_len = 0

        if self.search_clue_seg:
            clue_prompt_token = self.tokenizer(
                clue_prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            # zecheng_note: 这里要定位出关键信息的位置
            assert len(clue_list) > 0, "clue_list should not be empty if search_clue_seg is True"
            clue_pos = []
            prompt_len = prompt_token.input_ids.size(-1)
            # print(f"prompt_token.input_ids shape {prompt_token.input_ids.shape}")
            for clue in clue_list:
                clue_ids = self.tokenizer(clue, return_tensors="pt", add_special_tokens=False)["input_ids"]
                clue_ids = clue_ids[0, 1:-1]  # 去掉开头和结尾的token做匹配
                # print(f"clue_ids shape: {clue_ids.shape}")
                clue_len = clue_ids.size(-1)
                for i in range(prompt_len - clue_len + 1):
                    if torch.equal(prompt_token.input_ids[0, i : i + clue_len], clue_ids):
                        clue_pos.append((i, i + clue_len - 1))
                        break 

            clue_prompt_ids_len = clue_prompt_token["attention_mask"].int().sum().item()

        # filter the sample whose length is greater than max_length (2 for answer length)

        if not prompt or not response or prompt_ids_len >= self.max_length - 2:
            prompt = None

        if self.search_clue_seg and (len(clue_pos) != len(clue_list)):
            prompt = None

        if self.search_clue_seg:
            return {"prompt": prompt, "clue_prompt": clue_prompt, "response": response, "prompt_ids_len": prompt_ids_len, "clue_prompt_ids_len": clue_prompt_ids_len, "clue_pos": clue_pos}
        
        return {"prompt": prompt, "clue_prompt": None, "response": response, "prompt_ids_len": prompt_ids_len, "clue_prompt_ids_len": None, "clue_pos": None}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if self.clue_prompts:
            clue_prompt = self.clue_prompts[idx]
            clue_prompt_ids_len = self.clue_prompt_ids_lens[idx]
            clue_pos = self.clue_poss[idx]
        else:
            clue_prompt = None
            clue_prompt_ids_len = None
            clue_pos = None

        text = (prompt + response).rstrip("\n")
        if clue_prompt: 
            clue_text = (clue_prompt + response).rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
            if clue_prompt:
                clue_text += " " + self.tokenizer.eos_token

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if clue_prompt:
            clue_input_token = self.tokenizer(
                clue_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            clue_input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            clue_input_token["attention_mask"][0][-1] = True
            extra_info = {"clue_input": clue_text, "clue_input_length": clue_input_token["attention_mask"].int().sum().item(), "clue_pos": clue_pos}
        else:
            clue_input_token = None
            extra_info = None
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item()}
        if extra_info:
            info.update(extra_info)
            return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info, clue_prompt_ids_len, clue_input_token["input_ids"], clue_input_token["attention_mask"]
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info, None, None, None

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info, _, _, _ in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos, None, None, None


    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_clue_input_ids = []
        packed_attention_masks = []
        packed_clue_attention_masks = []
        prompt_ids_lens = []
        clue_prompt_ids_lens = []
        infos = {"input_length": [], "clue_input_length": [], "clue_poss": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info, pack_prompt_ids_len, clue_input_id, _ in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])

            if pack_prompt_ids_len is not None:
                packed_clue_input_ids.append(clue_input_id.flatten())
                packed_clue_attention_masks.append(torch.full_like(clue_input_id.flatten(), index))
                clue_prompt_ids_lens.append(pack_prompt_ids_len)
                infos["clue_input_length"].append(info["clue_input_length"])
                infos["clue_poss"].append(info["clue_pos"])
                
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        if self.search_clue_seg:
            packed_clue_input_ids = torch.cat(packed_clue_input_ids, dim=0).unsqueeze(0)
            packed_clue_attention_masks = torch.cat(packed_clue_attention_masks, dim=0).unsqueeze(0)

            if self.multiple_of > 1:  # not divisible by multiple_of; here we align for grouping
                padding_len = self.multiple_of - (packed_clue_input_ids.numel() % self.multiple_of)
                packed_clue_input_ids = F.pad(packed_clue_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
                packed_clue_attention_masks = F.pad(packed_clue_attention_masks, (0, padding_len), value=0)

        if (
            self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
        ):  # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)
        
        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos, clue_prompt_ids_lens, packed_clue_input_ids, packed_clue_attention_masks