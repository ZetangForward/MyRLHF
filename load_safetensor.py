from datasets import load_dataset

for split in ['0k', '2k', '4k', '8k', '16k', '32k']:
    content = load_dataset("RMT-team/babilong-train-5k-samples", split)