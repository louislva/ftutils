import openai, os, time, argparse
from ftutils.conversation import Conversation, Dataset, Message
import tiktoken
import argparse
import random
import hashlib

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name != "base.txt":
                r.append(os.path.join(root, name))
    return r

def estimate_tokens(param):
    if isinstance(param, str):
        if os.path.exists(param):
            if param.endswith(".jsonl"):
                return Dataset.from_file(param).tokens
            else:
                return Conversation.from_file(param).tokens
        else:
            return Conversation.from_text(param).tokens
    elif isinstance(param, Dataset):
        return param.tokens
    elif isinstance(param, Conversation):
        return param.tokens
    elif isinstance(param, Message):
        return param.tokens

def openai_start_finetune():
    parser = argparse.ArgumentParser(description="Start an OpenAI fine-tuning job.")
    parser.add_argument('-t', '--train', help='Train Dataset (JSONL format)', required=True)
    parser.add_argument('-e', '--eval', help='Eval Dataset (JSONL format)', required=False)
    parser.add_argument('-n', '--epochs', help='Number of epochs', default=3, type=int)
    parser.add_argument('-m', '--model', help='Base model', default='gpt-3.5-turbo-0613', type=str)
    args = parser.parse_args()

    root_dir = os.getcwd()

    print()
    print("Base model:", args.model)
    tokens = estimate_tokens(root_dir + "/" + args.train) + (estimate_tokens(root_dir + "/" + args.eval) if args.eval is not None else 0)
    cost = tokens * (0.008 / 1000) * args.epochs
    print("Estimated Tokens (train):", tokens)
    print("Estimated Cost (train): $" + str(round(cost, 2)))
    print()

    confirmation = input("Do you want to start the training run? (y/N): ")
    if confirmation.lower().strip() != 'y':
        print("Training run cancelled.")
        exit()

    print("Uploading file(s)...")
    oa_file = openai.files.create(
        file=open(root_dir + "/" + args.train, "rb"),
        purpose='fine-tune'
    )
    oa_file_eval = openai.files.create(
        file=open(root_dir + "/" + args.eval, "rb"),
        purpose='fine-tune'
    ) if args.eval is not None else None
    print("Created file(s) successfully:", oa_file.id, oa_file_eval.id if oa_file_eval is not None else '')
    print("Waiting for 30s...")
    time.sleep(30)
    print("Creating fine-tuning job...")
    fine_tune = openai.fine_tuning.jobs.create(training_file=oa_file.id, validation_file=(oa_file_eval.id if oa_file_eval is not None else None), model=args.model, hyperparameters={"n_epochs": args.epochs})
    print("Created fine tuning job:", fine_tune.id)

def openai_create_dataset():
    parser = argparse.ArgumentParser(description="Combine directories in conversations/ into a single file, for use in OpenAI fine-tuning.")
    parser.add_argument('-i', '--input', nargs='+', help='Input directories, inside conversations/', required=True)
    parser.add_argument('-o', '--output', help='Output file (JSONL format)', required=True)
    parser.add_argument('-s', '--split', type=float, help='Fraction of data to put into validation set randomly', default=0.0)
    args = parser.parse_args()

    root_dir = os.getcwd()

    paths = []
    for input in args.input:
        paths.extend(list_files(f"{root_dir}/conversations/{input}"))
    conversations = [Conversation.from_file(path) for path in paths]

    # This is a way of randomly, but deterministically, splitting the conversations into train/eval sets
    # This means that even if you do a different combination of sources, or add more files, things will stay either in train or eval
    keys = [float(int(hashlib.sha256(
        os.path.basename(path).encode('utf-8')
    ).hexdigest()[:4], 16) / (16 ** 4)) for path in paths]    
    conversations_train = [convo for key, convo in zip(keys, conversations) if key > args.split]
    conversations_eval = [convo for key, convo in zip(keys, conversations) if key <= args.split]
    
    dataset_train = Dataset(conversations_train)
    dataset_train.to_file(f"{root_dir}/datasets/{args.output}.train.jsonl")
    if len(conversations_eval) > 0:
        dataset_eval = Dataset(conversations_eval)
        dataset_eval.to_file(f"{root_dir}/datasets/{args.output}.eval.jsonl")