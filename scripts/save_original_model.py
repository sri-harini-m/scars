#scripts/save_original_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import load_config

paths = load_config("configs/paths.yaml")
BASE_MODEL = paths["original_checkpoint"]

model=AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tok=AutoTokenizer.from_pretrained(BASE_MODEL, torch_dtype="bfloat16")

model.save_pretrained(
    "checkpoints/original"
)

tok.save_pretrained(
    "checkpoints/original"
)