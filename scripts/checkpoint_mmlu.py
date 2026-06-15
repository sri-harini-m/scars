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

def evaluate_checkpoint(ckpt):
    ckpt_tag = ckpt.replace("/", "_").replace("\\", "_")
    output_path = f"/tmp/mmlu_{ckpt_tag}.json"

    if os.path.exists(output_path):
        print(f" [cached] {ckpt}")
    else:
        print(f" [running] {ckpt}")
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={ckpt}",
            "--tasks", "mmlu",
            "--batch_size", "auto",
            "--output_path", output_path
        ]
        subprocess.run(cmd, check=True)

    with open(output_path) as f:
        data = json.load(f)

    return data["results"]["mmlu"]["acc,none"]

def collect_checkpoints(root_dirs):
    entries = [] 
    for root_dir in root_dirs:
        run_name = os.path.basename(root_dir)

        step_ckpts = sorted(
            glob.glob(os.path.join(root_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1])
        )
        for ckpt in step_ckpts:
            entries.append((run_name, ckpt))

        if os.path.exists(os.path.join(root_dir, "config.json")):
            entries.append((run_name, root_dir))

    return entries

def main():
    os.makedirs("results/mmlu", exist_ok=True)
    entries = collect_checkpoints(CHECKPOINT_DIRS)

    if not entries:
        print(
            "No checkpoints found."
            "Please run train.py with degredation, recovery, and control configs first."
        )
        return
    
    results = []
    for run_name, ckpt in entries:
        print(f"\nEvaluating [{run_name}] {ckpt}")
        score = evaluate_checkpoint(ckpt)
        results.append({
            "run": run_name,
            "checkpoint": ckpt,
            "mmlu": score,
        })
        print(f"  MMLU: {score:.4f}")
 
    df = pd.DataFrame(results)
    csv_path = "results/mmlu/all_checkpoints.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.to_string())

    recovered_rows = df[df["run"] == "recovered"].copy()
 
    if recovered_rows.empty:
        print("\nNo recovered checkpoints found. Run configs/recovery.yaml first.")
        return
    
    thetar_row = recovered_rows.iloc[-1]
    thetar_path = thetar_row["checkpoint"]
    thetar_mmlu = thetar_row["mmlu"]
    print(f"\ntheta_r: {thetar_path}  (MMLU={thetar_mmlu:.4f})")

    control_rows = df[df["run"] == "control"].copy()
 
    if control_rows.empty:
        print(
            "\nNo control checkpoints found. "
            "Run train.py --config configs/control.yaml first, "
            "then re-run this script."
        )

        with open("results/mmlu/theta_r.json", "w") as f:
            json.dump({
                "theta_r": thetar_path,
                "theta_r_mmlu": float(thetar_mmlu),
            }, f, indent=2)
        return
 
    control_rows["mmlu_diff"] = (control_rows["mmlu"] - thetar_mmlu).abs()
    theta_ft_row = control_rows.loc[control_rows["mmlu_diff"].idxmin()]
    theta_ft_path = theta_ft_row["checkpoint"]
    theta_ft_mmlu = theta_ft_row["mmlu"]
    mmlu_diff = theta_ft_row["mmlu_diff"]
 
    print(
        f"theta_ft: {theta_ft_path}"
        f"  (MMLU={theta_ft_mmlu:.4f}, |diff from theta_r|={mmlu_diff:.4f})"
    )
 
    if mmlu_diff > 0.02:
        print(
            f"\nWARNING: best-matched control checkpoint is {mmlu_diff:.4f} MMLU "
            "away from theta_r. Consider running control training longer / with "
            "more checkpoint saves (lower save_steps) to get a tighter match."
        )

    summary = {
        "theta_r": thetar_path,
        "theta_r_mmlu": float(thetar_mmlu),
        "theta_ft": theta_ft_path,
        "theta_ft_mmlu": float(theta_ft_mmlu),
        "mmlu_diff": float(mmlu_diff),
    }
    summary_path = "results/mmlu/theta_ft.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")
 
    print("\n--- Next steps ---")
    print(
        f"python scripts/compute_rob.py "
        f"--target {thetar_path} --tag recovered"
    )
    print(
        f"python scripts/compute_rob.py "
        f"--target {theta_ft_path} --tag control"
    )
    print("python scripts/weight_distance.py")
    print("python scripts/trajectory_pca.py")
 
 
if __name__ == "__main__":
    main()
 

    
