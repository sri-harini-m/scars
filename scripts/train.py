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

# Llama's eos token id is 128009 - a valid vocabulary token.
# We must NOT use it as pad_token because when the collator pads labels
# it would insert real token ids that the loss function then tries to
# predict, causing the "t >= 0 && t < n_classes" assertion - the loss
# sees a label that looks valid but is in a padding position with no
# corresponding input context.
#
# Instead add a dedicated pad token that sits outside the vocab, or
# use unk_token if available. For Llama we add a new pad token.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # Resize model embeddings to account for the new token - done after
    # model load below

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
    device_map="auto",
    attn_implementation="eager",
)

# Resize embeddings if we added a pad token
if tokenizer.pad_token == "<|pad|>":
    model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing to reduce activation memory
model.gradient_checkpointing_enable()

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
    # Gradient checkpointing is set on the model above; this tells
    # Trainer not to override it
    gradient_checkpointing=False,
)

# label_pad_token_id=-100 is critical: the collator will replace all
# padding positions in the labels tensor with -100, which tells
# CrossEntropyLoss to ignore those positions entirely. Without this,
# padding token ids end up in labels and the loss tries to predict them.
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