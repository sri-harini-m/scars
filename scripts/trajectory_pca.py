#scripts/trajectory_pca.py

import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

N_PARAMS = 1_000_000

def load_model_vector(model_path, indices=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    vec = torch.cat(
        [
            p.detach()
            .float()
            .flatten()
            .cpu()
            for p in model.parameters()
        ]
    )

    if indices is None:
        torch.manual_seed(42)
        indices = torch.randperm(
            len(vec)
        )[:N_PARAMS]
    sampled = vec[indices]
    del model
    return sampled.numpy(), indices

def collect_checkpoints():
    entries = [] 

    if os.path.exists("checkpoints/original"):
        entries.append(("original", "checkpoints/original"))    
    
    for p in sorted(glob.glob("checkpoints/degraded/checkpoint-*")):
        entries.append(("degraded", p))

    for p in sorted(glob.glob("checkpoints/recovered/checkpoint-*")):
        entries.append(("recovered", p))

    for p in sorted(glob.glob("checkpoints/control/checkpoint-*")):
        entries.append(("control", p))

    return entries    


def main():
    os.makedirs("results/pca", exist_ok=True)

    entries = collect_checkpoints()
    if not entries:
        print("No checkpoints found in checkpoints/original, checkpoints/degraded, checkpoints/recovered, or checkpoints/control")
        print("Please run the training script to generate checkpoints first.")
        return

    print(f"Found {len(entries)} checkpoints across all runs")

    vectors = []
    indices = None

    for label, path in tqdm(entries, desc="Loading checkpoints"):
        vec, indices = load_model_vector(path, indices)
        vectors.append(vec)

    vectors = np.stack(vectors)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    labels = [e[0] for e in entries]
    colors = {
        "original": "black",
        "degraded": "red",
        "recovered": "blue",
        "control": "green",
    }
    markers = {
        "original": "*",
        "degraded": "o",
        "recovered": "s",
        "control": "^",
    }

    plt.figure(figsize=(9,7))

    original_indices = [i for i, l in enumerate(labels) if l == "original"]
    for run in ["degraded", "recovered", "control"]:
        idx = [i for i, l in enumerate(labels) if l == run]
        if not idx:
            continue
        path_idx = (original_indices + idx) if run == "degraded" else idx

        xs = coords[path_idx, 0]
        ys = coords[path_idx, 1]
        plt.plot(
            xs, ys,
            color=colors[run],
            marker=markers[run],
            label=run,
            linewidth=1.5,
            markersize=5,
        )

    for i, l in enumerate(labels):
        if l == "original":
            plt.scatter(
                coords[i, 0], coords[i, 1],
                color="black", marker="*", s=200, zorder=5, label="theta_0"
            )

    plt.legend()
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"
    )
    plt.title("Weight-space trajectories (PCA)")
    plt.tight_layout()
    plt.savefig("results/pca/trajectory.png", dpi=300)
    plt.close()

    np.save("results/pca/coords.npy", coords)
    np.save("results/pca/labels.npy", np.array(labels))

    print("Saved results/pca/trajectory.png and coords.npy")
    print(
        f"Variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
    )


if __name__ == "__main__":
    main()