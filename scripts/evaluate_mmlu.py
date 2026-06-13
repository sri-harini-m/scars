# scripts/evaluate_mmlu.py

import argparse
import subprocess
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True
)

parser.add_argument(
    "--output",
    default=None
)

args = parser.parse_args()

os.makedirs(
    "results/mmlu",
    exist_ok=True
)

if args.output is None:

    model_name = (
        args.model
        .replace("/", "_")
        .replace("\\", "_")
    )

    args.output = (
        f"results/mmlu/{model_name}.json"
    )

cmd = [
    "lm_eval",
    "--model", "hf",
    "--model_args",
    f"pretrained={args.model}",
    "--tasks", "mmlu",
    "--batch_size", "auto",
    "--output_path",
    args.output,
]

print("Running:")
print(" ".join(cmd))

subprocess.run(
    cmd,
    check=True
)

print(f"\nSaved results to: {args.output}")