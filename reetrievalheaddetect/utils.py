import emoji
import random
from transformers import AutoTokenizer


def get_random_emoji(tokenizer, num=50, return_idx=True):
    all_emojis = list(emoji.EMOJI_DATA.keys())  # get all emojis
    random_emojis = random.sample(all_emojis, num)
    print("your chose emoji:")
    print(random_emojis)

    if return_idx:
        index_emojis = []
        for e in random_emojis:
            index_emojis.append(tokenizer(e, add_special_tokens=False).input_ids)
        return index_emojis

    return random_emojis



if __name__ == "__main__":
    # 加载Llama的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    res = get_random_emoji(tokenizer, 50, True)
    print(res)