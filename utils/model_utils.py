#utils/model_utils.py

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

def load_model(model_path, bf16=True):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer