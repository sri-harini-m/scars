#scripts/save_original_model.py

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import load_config

SAVE_PATH = "checkpoints/original"

if os.path.exists(SAVE_PATH):
    print(
        f"ERROR: {SAVE_PATH} already exists. "
        "Delete it explicitly if you intend to reset theta_0. "
        "Aborting to avoid accidentally overwriting a checkpoint "
        "that degradation/recovery runs were already computed against."
    )
    sys.exit(1)

cfg = load_config("configs/paths.yaml")
name = cfg["original_model"]

print(f"Saving {name} -> {SAVE_PATH}")

model=AutoModelForCausalLM.from_pretrained(name)
tok=AutoTokenizer.from_pretrained(name, torch_dtype="bfloat16")

model.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)

print(f"Done. theta_0 saved to {SAVE_PATH}")
print(
    "\nPipeline order from here:\n"
    "  1. train.py --config configs/degradation.yaml\n"
    "  2. train.py --config configs/recovery.yaml\n"
    "  3. train.py --config configs/control.yaml\n"
    "  4. checkpoint_mmlu.py   (tracks scores; identifies theta_ft)\n"
    "  5. compute_rob.py       (ROB for theta_r vs theta_0)\n"
    "  6. compute_control_rob.py (ROB for theta_ft vs theta_0)\n"
    "  7. trajectory_pca.py\n"
)
