import argparse
import os
from src.conversation import Conversation, Dataset
import random
import hashlib

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name != "base.txt":
                r.append(os.path.join(root, name))
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine directories in conversations/ into a single file, for use in OpenAI fine-tuning.")
    parser.add_argument('-i', '--input', nargs='+', help='Input directories, inside conversations/', required=True)
    parser.add_argument('-o', '--output', help='Output file (JSONL format)', required=True)
    parser.add_argument('-s', '--split', type=float, help='Fraction of data to put into validation set randomly', default=0.0)
    args = parser.parse_args()

    root_dir = os.path.dirname(__file__)

    paths = []
    for input in args.input:
        paths.extend(list_files(f"{root_dir}/conversations/{input}"))
    conversations = [Conversation.from_file(path) for path in paths]

    # This is a way of randomly, but deterministically, splitting the conversations into train/eval sets
    # This means that even if you do a different combination of sources, or add more files, things will stay either in train or eval
    keys = [float(int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:4], 16) / (16 ** 4)) for path in paths]    
    conversations_train = [convo for key, convo in zip(keys, conversations) if key > args.split]
    conversations_eval = [convo for key, convo in zip(keys, conversations) if key <= args.split]
    
    dataset_train = Dataset(conversations_train)
    dataset_train.to_file(f"{root_dir}/datasets/{args.output}.train.jsonl")
    if len(conversations_eval) > 0:
        dataset_eval = Dataset(conversations_eval)
        dataset_eval.to_file(f"{root_dir}/datasets/{args.output}.eval.jsonl")