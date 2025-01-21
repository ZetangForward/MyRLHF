import emoji
import random
from transformers import AutoTokenizer




if __name__ == "__main__":
    # 加载Llama的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    res = get_random_emoji(tokenizer, 50, True)
    print(res)