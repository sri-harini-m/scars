# utils/dataset_utils.py

from datasets import load_from_disk

def format_codealpaca(example):
    messages = [
        {
            "role": "user",
            "content": example["prompt"]
        },
        {
            "role": "assistant",
            "content": example["completion"]
        }
    ]

    return {
        "messages": messages
    }

def format_openhermes(example):
    messages = []

    for turn in example["conversations"]:

        if turn["from"] == "human":

            messages.append({
                "role": "user",
                "content": turn["value"]
            })

        elif turn["from"] == "gpt":

            messages.append({
                "role": "assistant",
                "content": turn["value"]
            })

    return {
        "messages": messages
    }

def load_training_dataset(dataset_name):
    if dataset_name == "codealpaca":
        ds = load_from_disk(
            "data/codealpaca"
        )
        ds = ds.map(
            format_codealpaca
        )
        return ds

    elif dataset_name == "openhermes":
        ds = load_from_disk(
            "data/openhermes"
        )
        ds = ds.map(
            format_openhermes
        )
        return ds
    else:
        raise ValueError(dataset_name)