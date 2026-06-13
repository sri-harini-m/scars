# scripts/train.py

import torch 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from utils.config import load_config
from utils.dataset_utils import load_training_dataset
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    required=True,
)

args = parser.parse_args()
cfg = load_config(args.config)

print("Loading dataset...")
dataset = load_training_dataset(
    cfg["dataset"]
)
print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    cfg["model_name"]
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizing...")

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )
    result = tokenizer(
        text,
        truncation=True,
        max_length=cfg["max_length"],
        padding=False
    )
    result["labels"] = result["input_ids"].copy()

    return result

dataset = dataset.map(
    tokenize, 
    batched=True,
    remove_columns=dataset.column_names,
    load_from_cache_file=False
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    torch_dtype=torch.float16,
)

training_args = TrainingArguments(
    output_dir=cfg["output_dir"],
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    bf16=True,
    logging_steps=20,
    save_total_limit=None,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()

trainer.save_model(
    f"{cfg['output_dir']}/final"
)

tokenizer.save_pretrained(
    f"{cfg['output_dir']}/final"
)