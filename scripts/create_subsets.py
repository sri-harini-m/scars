from datasets import load_from_disk

ds = load_from_disk(
    "data/openhermes"
)

small = ds["train"].shuffle(
    seed=42
).select(
    range(50000)
)

small.save_to_disk(
    "data/openhermes"
)