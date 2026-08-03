# scripts/train.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from utils.config import load_config
from utils.dataset_utils import load_training_dataset
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)

print("Loading dataset...")
dataset = load_training_dataset(cfg["dataset"])
print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

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
        padding=False,  
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = dataset["train"].map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
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
    save_total_limit=2,
    save_only_model=True,
    logging_steps=20,
    report_to="none",
    no_cuda=False,
    fsdp="",                          
    dataloader_pin_memory=False,     
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

trainer.train()

trainer.save_model(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])

print(f"\nDone. Model saved to {cfg['output_dir']}")