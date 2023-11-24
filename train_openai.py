import openai, os, time, argparse
from src.conversation import Conversation, Dataset
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
def estimate_tokens(path):
    tokens = 0
    dataset = Dataset.from_file(path)
    for conversation in dataset.conversations:
        for message in conversation.messages:
            tokens += 2 # every message has 2 special tokens
            tokens += len(encoding.encode(message.content)) # correct way
            # tokens += len(message.content) / 4 # fast way
    return int(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start an OpenAI fine-tuning job.")
    parser.add_argument('-t', '--train', help='Train Dataset (JSONL format)', required=True)
    parser.add_argument('-e', '--eval', help='Eval Dataset (JSONL format)', required=False)
    parser.add_argument('-n', '--epochs', help='Number of epochs', default=3, type=int)
    parser.add_argument('-m', '--model', help='Base model', default='gpt-3.5-turbo-1106', type=str)
    args = parser.parse_args()

    root_dir = os.path.dirname(__file__)

    tokens = estimate_tokens(root_dir + "/" + args.train)
    cost = tokens * (0.008 / 1000) * args.epochs
    print("Estimated Tokens (train):", tokens)
    print("Estimated Cost (train): $" + str(round(cost, 2)))

    confirmation = input("Do you want to run the training run? (y/N): ")
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
    fine_tune = openai.fine_tuning.jobs.create(training_file=oa_file.id, validation_file=oa_file_eval, model=args.model, hyperparameters={"n_epochs": args.epochs})
    print("Created fine tuning job:", fine_tune.id)