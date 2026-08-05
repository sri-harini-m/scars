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

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)

print("Loading dataset...")
dataset = load_training_dataset(cfg["dataset"])
print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizing...")

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
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

print(f"Train dataset size: {len(train_dataset)}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

print(model.is_gradient_checkpointing)
print(model.config.use_cache)

# if len(tokenizer) != model.config.vocab_size:
#     model.resize_token_embeddings(len(tokenizer)) 
#     print(f"Resized embeddings to {len(tokenizer)}")

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
    deepspeed="configs/ds_zero3.json",
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

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}: "
        f"{torch.cuda.memory_allocated(i)/1024**3:.2f} GB allocated"
    )

trainer.train()

trainer.save_model(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])

print(f"\nDone. Model saved to {cfg['output_dir']}")