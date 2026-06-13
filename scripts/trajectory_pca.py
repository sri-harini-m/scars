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


def main():
    os.makedirs(
        "results/pca",
        exist_ok=True
    )
    checkpoints = []
    checkpoints.append("checkpoints/original")

    checkpoints.extend(
        sorted(
            glob.glob(
                "checkpoints/degraded/checkpoint-*"
            )
        )
    )

    checkpoints.extend(
        sorted(
            glob.glob(
                "checkpoints/recovered/checkpoint-*"
            )
        )
    )

    print(f"{len(checkpoints)} checkpoints")

    vectors = []

    indices = None

    for ckpt in tqdm(checkpoints):
        vec, indices = load_model_vector(
            ckpt,
            indices
        )
        vectors.append(vec)

    vectors = np.stack(vectors)

    pca = PCA(n_components=2)

    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(8,6))

    n_deg = len(
        glob.glob(
            "checkpoints/degraded/checkpoint-*"
        )
    )

    plt.plot(
        coords[:n_deg+1,0],
        coords[:n_deg+1,1],
        marker="o",
        label="degradation"
    )

    plt.plot(
        coords[n_deg:,0],
        coords[n_deg:,1],
        marker="s",
        label="recovery"
    )

    plt.legend()

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title("Weight-space trajectory")

    plt.savefig(
        "results/pca/trajectory.png",
        dpi=300
    )

    np.save(
        "results/pca/coords.npy",
        coords
    )


if __name__ == "__main__":
    main()