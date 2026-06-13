#scripts/download_datasets.py

from datasets import load_dataset
import os 

os.makedirs("data", exist_ok=True)

print("Downloading CodeAlpaca...")
codealpaca = load_dataset("HuggingFaceH4/CodeAlpaca_20K")
codealpaca.save_to_disk("data/codealpaca")

print("Downloading OpenHermes...")
openhermes = load_dataset("teknium/OpenHermes-2.5")
openhermes.save_to_disk("data/openhermes")

# print("Downloading WikiText...")
# wikitext = load_dataset(
#     "wikitext",
#     "wikitext-103-v1"
# )
# wikitext.save_to_disk("data/wikitext103")

print("Done.")