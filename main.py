###
# Toy Experiment for applying DPO for model editing
# Dataset: counterfact_sample.json
# Model: llama2-chat
# Experiment: perform single edit on model with default DPO settings and compare to other models
###

import argparse
import json
import os

import torch
from trl import DPOTrainer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments
)
from datasets import Dataset

from edits_as_preferences.dataset import (
    counterfact_preferences_ds,
)
from edits_as_preferences.evaluate import evaluate_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='distilgpt2')
argparser.add_argument('--sample-path', type=str, default='data/counterfact_samples')
argparser.add_argument('--sample-number', type=int, default=0)
argparser.add_argument('--max-steps', type=int, default=100)
argparser.add_argument('--beta', type=float, default=0.1)
argparser.add_argument('--include-paraphrases', action='store_true')
argparser.add_argument('--include-generations', action='store_true')
argparser.add_argument('--include-locality-control', action='store_true')
argparser.add_argument('--experiment-name', type=str, default='default')

args = argparser.parse_args()


if __name__ == '__main__':
    args = argparser.parse_args()
    print(f"args used: {args}")

    ## Construct dataset from edit sample
    print(f"Loading sample {args.sample_number} from {args.sample_path}")
    sample_file_path = f"{args.sample_path}/generated_{args.sample_number}.json"
    with open(sample_file_path, 'r') as f:
        edit_sample = json.load(f)

    setting = ['single']
    if args.include_paraphrases:
        setting.append('paraphrases')
    if args.include_generations:
        setting.append('with_generations')
    if args.include_locality_control:
        setting.append('with_locality_control')
    dataset = counterfact_preferences_ds([edit_sample], setting=setting)

    ## Load model
    print(f"Loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model,  low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token # this is required for training
    pre_performance = evaluate_model(
        model,
        args.model,
        edit_sample,
        tokenizer,
        DEVICE
    )
    # Training arguments
    print("Training model with DPO")
    training_args = TrainingArguments(
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=args.max_steps,
        save_strategy="no",
        logging_steps=args.max_steps // 10,
        output_dir="./tmp",
        per_device_train_batch_size=1,
        warmup_steps=100
    )
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    dpo_trainer.train()
    post_performance = evaluate_model(
        model,
        args.model,
        edit_sample,
        tokenizer,
        DEVICE
    )
    performance = {
        "pre": pre_performance,
        "post": post_performance
    }
    # make directory for results based on experiment_name
    directory = f"results/{args.experiment_name}"
    os.makedirs(directory, exist_ok=True)

    with open(f"{directory}/performance_{args.sample_number}.json", 'w+') as f:
        json.dump(performance, f)
