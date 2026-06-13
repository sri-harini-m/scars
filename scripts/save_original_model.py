#scripts/save_original_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

name="meta-llama/Llama-3.1-8B-Instruct"

model=AutoModelForCausalLM.from_pretrained(name)
tok=AutoTokenizer.from_pretrained(name)

model.save_pretrained(
    "checkpoints/original"
)

tok.save_pretrained(
    "checkpoints/original"
)