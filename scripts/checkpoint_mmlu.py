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
        # output_path = (f"/tmp/mmlu_result.json")
        ckpt_tag = ckpt.replace("/", "_").replace("\\", "_")
        output_path = f"/tmp/mmlu_{ckpt_tag}.json"

        cmd = f"""
        lm_eval \
            --model hf \
            --model_args pretrained={ckpt} \
            --tasks mmlu \
            --batch_size auto \
            --output_path {output_path}
        """

        subprocess.run(cmd, shell=True, check=True)

        with open(output_path, "r") as f:
            data = json.load(f)

        score = data["results"]["mmlu"]["acc,none"]

        results.append(
            {
                "checkpoint": ckpt,
                "run": os.path.basename(root_dir),
                "mmlu": score
            }
        )


os.makedirs("results/mmlu", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/mmlu/all_checkpoints.csv", index=False)
print(df.to_string())

recovered= "checkpoints/recovered"
if os.path.exists(recovered):
    output_path = "/tmp/mmlu_thetar.json"
    cmd = (f"""an
        lm_eval \
        --model hf \
        --model_args pretrained={recovered} \
        --tasks mmlu \
        --batch_size auto \
        --output_path {output_path}"""
    )
    subprocess.run(cmd, shell=True, check=True)
    with open(output_path) as f:
        thetar_mmlu = json.load(f)["results"]["mmlu"]["acc,none"]
    print(f"\ntheta_r MMLU: {thetar_mmlu:.4f}")
else:
    recovered_rows = df[df["run"] == "recovered"]
    thetar_mmlu = recovered_rows["mmlu"].max()
    print(f"\ntheta_r MMLU (estimated from checkpoints): {thetar_mmlu:.4f}")

control_rows = df[df["run"] == "control"].copy()
if control_rows.empty:
    print(
        "No control checkpoints found. "
        "Run train.py --config configs/control.yaml first."
    )
else:
    control_rows["mmlu_diff"] = (
        control_rows["mmlu"] - thetar_mmlu
    ).abs()
    theta_ft_row = control_rows.loc[control_rows["mmlu_diff"].idxmin()]
    theta_ft_path = theta_ft_row["checkpoint"]
    theta_ft_mmlu = theta_ft_row["mmlu"]

    print(
        f"\ntheta_ft identified: {theta_ft_path}"
        f"  (MMLU={theta_ft_mmlu:.4f}, "
        f"diff from theta_r={theta_ft_row['mmlu_diff']:.4f})"
    )
    print(
        "\nNext step:\n"
        f"  python scripts/compute_control_rob.py --theta_ft {theta_ft_path}"
    )

    with open("results/mmlu/theta_ft.json", "w") as f:
        json.dump(
            {
                "theta_ft": theta_ft_path,
                "theta_ft_mmlu": float(theta_ft_mmlu),
                "thetar_mmlu": float(thetar_mmlu),
                "mmlu_diff": float(theta_ft_row["mmlu_diff"]),
            },
            f,
            indent=2,
        )
