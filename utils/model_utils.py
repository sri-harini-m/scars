#utils/model_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, bf16=True):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer