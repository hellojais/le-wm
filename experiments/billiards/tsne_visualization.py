"""
t-SNE visualization of LeWM billiards latent space.
Loads lewm_small_epoch_8_object.ckpt, samples 2000 frames from
billiards_expert_train.h5, extracts 32-dim embeddings, runs t-SNE,
and creates a 2x2 figure colored by various state properties.
"""

import os
import math
import numpy as np
import torch
import h5py
import hdf5plugin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

# ── paths ──────────────────────────────────────────────────────────────────
CKPT_PATH = Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
HDF5_PATH = Path.home() / ".stable_worldmodel" / "datasets" / "billiards_expert_train.h5"
if not HDF5_PATH.exists():
    # fallback: look in billiards-worldmodel sibling folder
    HDF5_PATH = (
        Path(__file__).parent.parent.parent.parent / "billiards-worldmodel" / "billiards_expert_train.h5"
    )
if not HDF5_PATH.exists():
    raise FileNotFoundError(f"Cannot find billiards_expert_train.h5. Tried:\n  {HDF5_PATH}")
OUTPUT_PATH = Path(__file__).parent / "tsne_billiards.png"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
BATCH_SIZE = 64
N_SAMPLES = 2000
TSNE_PERPLEXITY = 30
TSNE_ITERS = 1000
NEAR_POCKET_THRESHOLD = 30.0  # pixels


# ── load model ─────────────────────────────────────────────────────────────
def load_encoder(ckpt_path: Path, device: torch.device):
    print(f"Loading checkpoint: {ckpt_path}")
    # The checkpoint is a serialised model object (not a state-dict dict)
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    model.to(device)
    embed_dim = getattr(model, "embed_dim", "?")
    print(f"  Model type: {type(model).__name__}, embed_dim={embed_dim}")
    return model


# ── sample frames from HDF5 ────────────────────────────────────────────────
def sample_frames(hdf5_path: Path, n_samples: int):
    """
    Sample n_samples frames spread evenly across all episodes.
    Returns (frames_uint8 [N,H,W,3], states [N,10]).
    """
    print(f"Opening dataset: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        # dataset layout: pixels (N,H,W,3) uint8, state (N,10) float32
        images_ds = f["pixels"]
        states_ds = f["state"]
        total_frames = images_ds.shape[0]
        print(f"  Total frames: {total_frames}")

        # evenly-spaced indices
        indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
        # sort so HDF5 reads are sequential
        indices = np.sort(indices)

        frames = images_ds[indices]   # (N,H,W,3) uint8
        states = states_ds[indices]   # (N,10) float32

    return frames, states, indices


# ── extract embeddings ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, frames_uint8, device, batch_size=64):
    """
    frames_uint8: (N, H, W, 3) uint8 numpy array
    Returns embeddings: (N, embed_dim) numpy array
    """
    # normalise to [0,1] and convert to (N, 3, H, W)
    frames_f = frames_uint8.astype(np.float32) / 255.0
    frames_t = torch.from_numpy(frames_f).permute(0, 3, 1, 2)  # (N,3,H,W)

    all_embs = []
    n = len(frames_t)
    for start in range(0, n, batch_size):
        batch = frames_t[start : start + batch_size].to(device)
        # model.encode expects {"pixels": (B, T, C, H, W)} — add T=1
        out = model.encode({"pixels": batch.unsqueeze(1)})
        emb = out["emb"]           # (B, 1, D)
        if emb.ndim == 3:
            emb = emb[:, 0, :]     # take the single token → (B, D)
        all_embs.append(emb.cpu().numpy())
        if start % (batch_size * 10) == 0:
            print(f"  Encoded {start}/{n} frames...")

    embeddings = np.concatenate(all_embs, axis=0)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


# ── state helpers ───────────────────────────────────────────────────────────
def extract_state_features(states):
    """
    State vector (10-dim):
    [cue_x, cue_y, cue_vx, cue_vy,
     target_x, target_y, target_vx, target_vy,
     nearest_pocket_x, nearest_pocket_y]
    """
    cue_x        = states[:, 0]
    cue_y        = states[:, 1]
    target_x     = states[:, 4]
    target_y     = states[:, 5]
    target_vx    = states[:, 6]
    target_vy    = states[:, 7]
    pocket_x     = states[:, 8]
    pocket_y     = states[:, 9]

    dist_to_pocket = np.sqrt(
        (target_x - pocket_x) ** 2 + (target_y - pocket_y) ** 2
    )
    return target_x, target_y, dist_to_pocket


# ── main ────────────────────────────────────────────────────────────────────
def main():
    # 1. Load model
    model = load_encoder(CKPT_PATH, DEVICE)

    # 2. Sample frames
    frames, states, indices = sample_frames(HDF5_PATH, N_SAMPLES)
    states = states.astype(np.float32)

    # 3. Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, frames, DEVICE, BATCH_SIZE)

    # 4. Compute state features
    target_x, target_y, dist_to_pocket = extract_state_features(states)
    emb_magnitude = np.linalg.norm(embeddings, axis=1)

    # 5. Run t-SNE
    print(f"Running t-SNE (perplexity={TSNE_PERPLEXITY}, n_iter={TSNE_ITERS})...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERS,
        random_state=42,
        verbose=1,
    )
    tsne_coords = tsne.fit_transform(embeddings)
    print(f"  t-SNE done. Shape: {tsne_coords.shape}")

    # 6. Near-pocket statistics
    near_mask = dist_to_pocket < NEAR_POCKET_THRESHOLD
    n_near = near_mask.sum()
    print(f"\nFrames where target near pocket (dist<{NEAR_POCKET_THRESHOLD}px): {n_near}")

    if n_near >= 2:
        near_coords = tsne_coords[near_mask]
        # mean pairwise distance in t-SNE space
        from scipy.spatial.distance import pdist
        mean_dist = pdist(near_coords).mean()
        print(f"Do near-pocket frames cluster? Mean distance between them in t-SNE: {mean_dist:.2f}")
    else:
        mean_dist = float("nan")
        print("Not enough near-pocket frames to compute clustering.")

    # 7. Plot 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("LeWM Billiards Latent Space (embed_dim=32)", fontsize=15, fontweight="bold")

    plot_configs = [
        (axes[0, 0], target_x,        "Target Ball X Position",    "RdYlGn",   "X (pixels)"),
        (axes[0, 1], target_y,        "Target Ball Y Position",    "RdYlGn",   "Y (pixels)"),
        (axes[1, 0], dist_to_pocket,  "Distance to Nearest Pocket","RdYlGn_r", "Distance (pixels)"),
        (axes[1, 1], emb_magnitude,   "Embedding Magnitude",       "viridis",  "L2 norm"),
    ]

    for ax, values, title, cmap, cbar_label in plot_configs:
        sc = ax.scatter(
            tsne_coords[:, 0], tsne_coords[:, 1],
            c=values, cmap=cmap, s=6, alpha=0.7,
            rasterized=True,
        )
        # highlight near-pocket frames
        if "Distance" in title and n_near > 0:
            ax.scatter(
                tsne_coords[near_mask, 0], tsne_coords[near_mask, 1],
                s=25, facecolors="none", edgecolors="black", linewidths=0.8,
                label=f"Near pocket (<{NEAR_POCKET_THRESHOLD}px, n={n_near})",
                zorder=5,
            )
            ax.legend(fontsize=8, loc="upper right")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(cbar_label, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE dim 1", fontsize=8)
        ax.set_ylabel("t-SNE dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
