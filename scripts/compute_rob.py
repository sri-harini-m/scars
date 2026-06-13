#scripts/compute_rob.py

import os
import json
import copy
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BASE_MODEL = "checkpoints/original"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_MODEL = "checkpoints/recovered/final"

N_POINTS = 30
MAX_SAMPLES = 512
SEQ_LEN = 512

def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model.eval()
    return model

def load_eval_dataset():
    ds = load_from_disk("data/openhermes_val")
    texts = []
    for x in ds["validation"]:
        txt = x["text"].strip()
        if len(txt) > 100:
            texts.append(txt)
        if len(texts) >= MAX_SAMPLES:
            break
    return texts

def evaluate_loss(model, tokenizer, texts):
    losses = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=SEQ_LEN,
                return_tensors="pt"
            )
            enc = {
                k: v.to(DEVICE)
                for k, v in enc.items()
            }
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
    return float(np.mean(losses))


def interpolate_model(model_a, model_b, alpha):
    interp = copy.deepcopy(model_a)
    with torch.no_grad():
        for p_i, p_a, p_b in zip(interp.parameters(), model_a.parameters(), model_b.parameters()):
            p_i.copy_((1 - alpha) * p_a + alpha * p_b)
    return interp


def main():
    os.makedirs(
        "results/rob",
        exist_ok=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    theta0 = load_model(BASE_MODEL).to(DEVICE)

    thetar = load_model(TARGET_MODEL).to(DEVICE)

    texts = load_eval_dataset()

    alphas = np.linspace(0, 1, N_POINTS)

    losses = []

    for alpha in tqdm(alphas):
        interp = interpolate_model(theta0, thetar, alpha)
        interp = interp.to(DEVICE)
        loss = evaluate_loss(interp, tokenizer, texts)
        losses.append(loss)
        del interp
        torch.cuda.empty_cache()

    rob = (max(losses)-max(losses[0], losses[-1]))

    plt.figure(figsize=(8,5))

    plt.plot(
        alphas,
        losses,
        marker="o"
    )

    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.title("ROB interpolation curve")

    plt.savefig(
        "results/rob/loss_curve.png",
        dpi=300
    )

    result = {
        "rob": float(rob),
        "max_loss": float(max(losses)),
        "loss_start": float(losses[0]),
        "loss_end": float(losses[-1])
    }

    with open("results/rob/rob.json", "w") as f:
        json.dump(result, f, indent=2)

    print(result)


if __name__ == "__main__":
    main()