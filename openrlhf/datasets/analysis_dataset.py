from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils import zero_pad_sequences


def preprocess_data(data, input_key="input", output_key=None, apply_chat_template=None):
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

    return prompt, response, data["all_clues"], data["attack_clues"]


class AnalysisDataset(Dataset):
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
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat template
        self.input_template = input_template
        self.input_key = "message"
        self.output_key = None
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        
        dataset = dataset.filter(lambda x: len(x['attack_clues']) > 0)

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors, 
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.attack_poss = processed_dataset["attack_pos"]
        self.clue_poss = processed_dataset["clue_pos"]

    def process_data(self, data):

        prompt, response, clue_list, attack_list = preprocess_data(
            data,
            self.input_key,
            self.output_key,
            self.apply_chat_template,
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

        # locate key information position and attack information position
        assert len(clue_list) > 0, "clue_list should not be empty"
        assert len(attack_list) > 0, "attack_list should not be empty"
        
        clue_pos, attack_pos = [], []
        prompt_len = prompt_token.input_ids.size(-1)

        for clue in clue_list:
            clue_ids = self.tokenizer(clue, return_tensors="pt", add_special_tokens=False)["input_ids"]
            clue_ids = clue_ids[0, 1:-1]  # remove begin and end tokens for exact matching
            clue_len = clue_ids.size(-1)
            for i in range(prompt_len - clue_len + 1):
                if torch.equal(prompt_token.input_ids[0, i : i + clue_len], clue_ids):
                    clue_pos.append((i, i + clue_len - 1))
                    break 

        for clue in attack_list:
            clue_ids = self.tokenizer(clue, return_tensors="pt", add_special_tokens=False)["input_ids"]
            clue_ids = clue_ids[0, 1:-1]  # remove begin and end tokens for exact matching
            clue_len = clue_ids.size(-1)
            for i in range(prompt_len - clue_len + 1):
                if torch.equal(prompt_token.input_ids[0, i : i + clue_len], clue_ids):
                    attack_pos.append((i, i + clue_len - 1))
                    break 

        # filter the sample whose length is greater than max_length (2 for answer length)
        if not prompt or not response or prompt_ids_len >= self.max_length - 2:
            prompt = None

        if (len(clue_pos) != len(clue_list)) or (len(attack_pos) != len(attack_list)):
            prompt = None
        
        return {
            "prompt": prompt, 
            "response": response, 
            "prompt_ids_len": prompt_ids_len, 
            "clue_pos": clue_pos, 
            "attack_pos": attack_pos
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        attack_pos = self.attack_poss[idx]
        clue_pos = self.clue_poss[idx]

        text = (prompt + response).rstrip("\n")
        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item()}
        info.update({"clue_pos": clue_pos, "attack_pos": attack_pos})
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": [], "clue_input_length": [], "clue_poss": [], "attack_poss": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            infos["clue_poss"].append(info["clue_pos"])
            infos["attack_poss"].append(info["attack_pos"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        if (
            self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
        ):  # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)
        
            
        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos