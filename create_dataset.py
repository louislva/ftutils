import argparse
import os
from src.conversation import Conversation, Dataset

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine directories in conversations/ into a single file, for use in OpenAI fine-tuning.")
    parser.add_argument('-i', '--input', nargs='+', help='Input directories, inside conversations/', required=True)
    parser.add_argument('-o', '--output', help='Output file (JSONL format)', required=True)
    args = parser.parse_args()

    root_dir = os.path.dirname(__file__)

    paths = []
    for input in args.input:
        paths.extend(list_files(f"{root_dir}/conversations/{input}"))
        
    dataset = Dataset([Conversation.from_file(path) for path in paths])
    dataset.to_file(f"{root_dir}/{args.output}")