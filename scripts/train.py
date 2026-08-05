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

# Add a dedicated pad token rather than reusing eos_token (128009).
# Reusing eos as pad causes the collator to insert real token ids into
# label padding positions, which triggers the nll_loss CUDA assert.
# This check handles the case where a previous training run already
# saved the tokenizer with <|pad|> added - we don't add it twice.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

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
        padding=False,  # collator handles per-batch padding
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
    attn_implementation="eager",
    # No device_map here - ZeRO-3 via torchrun handles device placement.
    # device_map="auto" conflicts with DDP/DeepSpeed and causes zero loss.
)

# Resize embeddings to account for the added <|pad|> token.
# Must happen before DeepSpeed wraps the model, so do it here.
# Only resize if vocab sizes are mismatched to avoid unnecessary work
# on subsequent runs where the tokenizer already has the token.
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)}")

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

# DataCollatorForSeq2Seq pads both input_ids and labels per batch,
# masking label padding positions with -100 so the loss ignores them.
# This is correct for causal LM - do not switch to DataCollatorWithPadding
# which does not handle labels.
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