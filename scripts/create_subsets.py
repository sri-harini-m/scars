from datasets import load_from_disk, DatasetDict

ds = load_from_disk(
    "data/openhermes"
)

small = ds["train"].shuffle(seed=42).select(range(50000))
DatasetDict({"train": small}).save_to_disk("data/openhermes_small")

print("saved")