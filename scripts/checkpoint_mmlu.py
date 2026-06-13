#scripts/checkpoint_mmlu.py

import os
import glob
import json
import pandas as pd
import subprocess

CHECKPOINT_DIRS = [
    "checkpoints/degraded",
    "checkpoints/recovered",
    "checkpoints/control"
]

results = []

for root_dir in CHECKPOINT_DIRS:
    checkpoints = sorted(
        glob.glob(os.path.join(root_dir, "checkpoint-*"))
    )

    if os.path.exists(os.path.join(root_dir, "config.json")):
        checkpoints.append(root_dir)

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt}")
        output_path = (f"/tmp/mmlu_result.json")

        cmd = f"""
        lm_eval \
            --model hf \
            --model_args pretrained={ckpt} \
            --tasks mmlu \
            --batch_size auto \
            --output_path {output_path}
        """

        subprocess.run(
            cmd,
            shell=True
        )

        with open(
            output_path,
            "r"
        ) as f:

            data = json.load(f)

        score = data["results"]["mmlu"][
            "acc,none"
        ]

        results.append(
            {
                "checkpoint": ckpt,
                "mmlu": score
            }
        )


df = pd.DataFrame(results)

df.to_csv(
    "results/mmlu/all_checkpoints.csv",
    index=False
)

print(df)