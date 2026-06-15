#scripts/compute_rob.py

import os
import json
import copy
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from utils.config import load_config
# from utils.dataset_utils import format_openhermes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_POINTS = 30
MAX_SAMPLES = 512
SEQ_LEN = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="path to target checkpoint (theta_r, theta_ft, etc.)")
    parser.add_argument("--tag", required=True, help="label for output files, e.g. 'recovered', 'control', 'degraded'")
    parser.add_argument("--base", default="checkpoints/original", help="path to base checkpoint (theta_0)")
    return parser.parse_args()

def load_model(path, device=DEVICE):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model

def load_eval_dataset():
    ds = load_from_disk("data/wikitext103")
    texts = []
    for x in ds["validation"]:
        txt = x["text"].strip()
        if len(txt) > 200:
            texts.append(txt)
        if len(texts) >= MAX_SAMPLES:
            break

    print(f"Loaded {len(texts)} evaluation texts from WikiText-103 validation")
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
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
    return float(np.mean(losses))


def interpolate_model(model_a, model_b, alpha):
    interp = copy.deepcopy(model_a)
    with torch.no_grad():
        for p_i, p_a, p_b in zip(
            interp.parameters(), 
            model_a.parameters(), 
            model_b.parameters()
        ):
            p_i.copy_((1 - alpha) * p_a + alpha * p_b)

    return interp

def compute_rob(losses, alphas):
    linear_baseline = [ 
        (1 - a) * losses[0] + a * losses[-1]
        for a in alphas
    ]
    barriers = [l - b for l, b in zip(losses, linear_baseline)]
    rob = max(barriers)
    return rob, linear_baseline, barriers


def main():
    args = parse_args()
    os.makedirs("results/rob", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.base}")
    theta0 = load_model(args.base, device=DEVICE)

    print(f"Loading target model from {args.target}")
    theta_target = load_model(args.target, device=DEVICE)

    texts = load_eval_dataset()
    print(f"Loaded {len(texts)} evaluation texts")
    
    alphas = np.linspace(0, 1, N_POINTS)
    losses = []

    for alpha in tqdm(alphas, desc=f"Interpolating {args.tag}"):
        interp = interpolate_model(theta0, theta_target, alpha)
        loss = evaluate_loss(interp, tokenizer, texts)
        losses.append(loss)
        del interp
        torch.cuda.empty_cache()

    rob, linear_baseline, barriers = compute_rob(losses, alphas)

    plt.figure(figsize=(9, 5))
    plt.plot(alphas, losses, marker="o", label="interpolation loss", zorder=3)
    plt.plot(
        alphas, linear_baseline,
        linestyle="--", color="gray", label="linear baseline"
    )
    plt.fill_between(
        alphas, linear_baseline, losses,
        alpha=0.2, color="red", label="barrier area"
    )
    plt.axvline(
        alphas[np.argmax(barriers)], color="red",
        linestyle=":", linewidth=1, label=f"peak barrier (ROB={rob:.4f})"
    )
    plt.xlabel(f"alpha  (0 = theta_0, 1 = {args.tag})")
    plt.ylabel("loss (WikiText-103 validation)")
    plt.title(f"ROB interpolation curve: theta_0 -> {args.tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/rob/loss_curve_{args.tag}.png", dpi=300)
    plt.close()
    print(f"Saved results/rob/loss_curve_{args.tag}.png")

    result = {
        "rob": float(rob),
        "max_loss": float(max(losses)),
        "loss_at_theta0": float(losses[0]),
        "loss_at_target": float(losses[-1]),
        "peak_barrier_alpha": float(alphas[np.argmax(barriers)]),
        "linear_baseline_at_peak": float(linear_baseline[np.argmax(barriers)]),
        "base_model": args.base,
        "target_model": args.target,
        "tag": args.tag,
        "eval_dataset": "wikitext103/validation",
        "n_eval_texts": len(texts),
        "n_interpolation_points": N_POINTS,
    }

    out_path = f"results/rob/rob_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")
    print(json.dumps(result, indent=2))
 
    control_path = f"results/rob/rob_control.json"
    if os.path.exists(control_path) and args.tag != "control":
        with open(control_path) as f:
            ctrl = json.load(f)
        control_rob = ctrl["rob"]
        rdf = rob / control_rob if control_rob > 0 else float("inf")

        rdf_result = {
            "rob": float(rob),
            "control_rob": float(control_rob),
            "rdf": rdf,
            "tag": args.tag,
        }

        rdf_path = f"results/rob/rdf_{args.tag}.json"
        with open(rdf_path, "w") as f:
            json.dump(rdf_result, f, indent=2)
 
        print(
            f"\nRDF ({args.tag}) = {rdf:.3f}"
            f"  (ROB={rob:.4f} / control_ROB={control_rob:.4f})"
        )
        if rdf > 1.0:
            print("-> Round-trip left a geometric scar beyond direct fine-tuning.")
        else:
            print("-> No measurable extra scar from the round-trip.")


if __name__ == "__main__":
    main()