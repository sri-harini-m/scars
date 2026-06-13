# scripts/create_rob_validation.py

from datasets import load_from_disk

ds = load_from_disk(
    "data/openhermes"
)

split = ds["train"].train_test_split(
    test_size=5000,
    seed=42
)

split["test"].save_to_disk(
    "data/openhermes_val"
)

print("saved")