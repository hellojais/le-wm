"""probe_framestacked.py — Extended probe diagnostic for the FrameStack model.

Option A diagnostic: PROBE_EPOCHS=1000, N_PROBE_FRAMES=5000 on the FrameStack
checkpoint only.  Reports position and velocity R² at convergence to determine
whether position information exists in the 32-dim embedding but wasn't
extractable with the 300-epoch / 1000-frame probe used in compare_predictors.py.

Run from le-wm/:
    uv run python experiments/billiards/probe_framestacked.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdf5plugin  # noqa: F401
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # le-wm/

# ── reuse helpers from compare_predictors ──────────────────────────────────
from experiments.billiards.compare_predictors import (
    CKPT_FRAMESTACKED,
    DATASET_PATH,
    DEVICE,
    STATE_NAMES,
    VEL_DIMS,
    POS_DIMS,
    load_model,
    build_episode_lookup,
    is_stacked_model,
    triplet_indices,
    preprocess,
)

# ── extended probe parameters ──────────────────────────────────────────────
PROBE_EPOCHS   = 1000
N_PROBE_FRAMES = 5000
PROBE_LR       = 1e-3
PROBE_BATCH    = 256
PROBE_HIDDEN   = 64
STATE_DIM      = 10
PROBE_SEED     = 42
LOG_EVERY      = 100


# ─────────────────────────────────────────────────────────────────────────────
# Probe model
# ─────────────────────────────────────────────────────────────────────────────

class StateProbe(nn.Module):
    def __init__(self, embed_dim: int, state_dim: int = STATE_DIM,
                 hidden: int = PROBE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, state_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_probe_data(model, f, frame_ep, frame_loc, ep_offset,
                       n_frames: int, seed: int):
    rng     = np.random.default_rng(seed)
    n_total = f["pixels"].shape[0]
    indices = np.sort(rng.choice(n_total, size=n_frames, replace=False))
    stacked = is_stacked_model(model)

    emb_list, state_list = [], []
    batch_size = 32
    for start in range(0, n_frames, batch_size):
        batch_idx = indices[start : start + batch_size]
        states_np = f["state"][batch_idx]

        if stacked:
            imgs = []
            for g in batch_idx:
                g   = int(g)
                ep  = int(frame_ep[g])
                loc = int(frame_loc[g])
                off = int(ep_offset[ep])
                g2, g1, g0 = triplet_indices(g, off, loc)
                p2, p1, p0 = (preprocess(f["pixels"][g2]),
                              preprocess(f["pixels"][g1]),
                              preprocess(f["pixels"][g0]))
                imgs.append(torch.cat([p2, p1, p0], dim=0))
            imgs_t = torch.stack(imgs).unsqueeze(0).to(DEVICE)
        else:
            frames_np = f["pixels"][batch_idx]
            imgs_t = torch.stack([preprocess(frames_np[j])
                                  for j in range(len(batch_idx))])
            imgs_t = imgs_t.unsqueeze(0).to(DEVICE)

        emb = model.encode({"pixels": imgs_t})["emb"][0, :len(batch_idx)]
        emb_list.append(emb.cpu())
        state_list.append(torch.from_numpy(states_np.astype(np.float32)))

    return torch.cat(emb_list, dim=0), torch.cat(state_list, dim=0)


def compute_r2(probe, embs_d, states_d):
    """R² per state dimension (no_grad)."""
    with torch.no_grad():
        p = probe(embs_d)
    ss_res = (p - states_d).pow(2).sum(0)
    ss_tot = (states_d - states_d.mean(0)).pow(2).sum(0).clamp(min=1e-8)
    return (1 - ss_res / ss_tot).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Option A diagnostic — extended probe on FrameStack embedding")
    print(f"  PROBE_EPOCHS={PROBE_EPOCHS}  N_PROBE_FRAMES={N_PROBE_FRAMES}")
    print("=" * 66)

    # ── dataset ──────────────────────────────────────────────────────────
    print(f"\n[0] Loading dataset: {DATASET_PATH.name}")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {len(ep_len):,} episodes, {f['pixels'].shape[0]:,} frames")

    print("    Building episode lookup …")
    frame_ep, frame_loc = build_episode_lookup(ep_len, ep_offset)
    print(f"    ✓ done")

    # ── model ─────────────────────────────────────────────────────────────
    print("\n[1] Loading FrameStack checkpoint")
    model = load_model(CKPT_FRAMESTACKED, "FrameStack")
    model.eval().requires_grad_(False)

    embed_dim = model.encode(
        {"pixels": torch.zeros(1, 1, 9, 96, 96, device=DEVICE)}
    )["emb"].shape[-1]
    print(f"    embed_dim = {embed_dim}")

    # ── extract embeddings ────────────────────────────────────────────────
    print(f"\n[2] Extracting {N_PROBE_FRAMES:,} frame embeddings …")
    embs, states = extract_probe_data(
        model, f, frame_ep, frame_loc, ep_offset, N_PROBE_FRAMES, seed=PROBE_SEED
    )
    print(f"    embs   : {tuple(embs.shape)}  (mean={embs.mean():.4f}, std={embs.std():.4f})")
    print(f"    states : {tuple(states.shape)}")

    # normalise targets
    state_mean = states.mean(0)
    state_std  = states.std(0).clamp(min=1e-6)
    states_n   = (states - state_mean) / state_std

    # ── probe training ────────────────────────────────────────────────────
    print(f"\n[3] Training probe ({PROBE_EPOCHS} epochs) …\n")
    probe    = StateProbe(embed_dim).to(DEVICE)
    opt      = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    embs_d   = embs.to(DEVICE)
    states_d = states_n.to(DEVICE)
    n        = len(embs_d)

    history = {}
    for epoch in range(PROBE_EPOCHS + 1):
        if epoch > 0:
            perm = torch.randperm(n, device=DEVICE)
            for bi in range(0, n, PROBE_BATCH):
                idx  = perm[bi : bi + PROBE_BATCH]
                loss = F.mse_loss(probe(embs_d[idx]), states_d[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        if epoch % LOG_EVERY == 0:
            r2 = compute_r2(probe, embs_d, states_d)
            vel_r2  = r2[VEL_DIMS].mean()
            pos_r2  = r2[POS_DIMS].mean()
            history[epoch] = r2.copy()
            per_dim = "  ".join(f"{STATE_NAMES[i]}={r2[i]:.3f}" for i in range(len(STATE_NAMES)))
            print(f"  epoch {epoch:4d}/{PROBE_EPOCHS}"
                  f"  vel_R²={vel_r2:.4f}  pos_R²(tgt)={pos_r2:.4f}")
            if epoch == PROBE_EPOCHS:
                print(f"\n  Per-dim: {per_dim}")

    # ── summary table ─────────────────────────────────────────────────────
    r2_300  = history.get(300,  history[min(history, key=lambda e: abs(e - 300))])
    r2_1000 = history[PROBE_EPOCHS]

    print("\n" + "─" * 66)
    print(f"  {'Dimension':<12}  {'R²@300ep':>9}  {'R²@1000ep':>9}  {'Δ':>7}")
    print("─" * 66)
    for i, name in enumerate(STATE_NAMES):
        tag = "[VEL]" if i in VEL_DIMS else "[POS]" if i in POS_DIMS else "     "
        delta = r2_1000[i] - r2_300[i]
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<12}  {r2_300[i]:9.4f}  {r2_1000[i]:9.4f}  {sign}{delta:.4f}  {tag}")
    print("─" * 66)

    v300  = r2_300[VEL_DIMS].mean()
    p300  = r2_300[POS_DIMS].mean()
    v1000 = r2_1000[VEL_DIMS].mean()
    p1000 = r2_1000[POS_DIMS].mean()
    print(f"  {'vel R² (mean)':<12}  {v300:9.4f}  {v1000:9.4f}  "
          f"{'+' if v1000-v300>=0 else ''}{v1000-v300:.4f}")
    print(f"  {'pos R² (tgt)':<12}  {p300:9.4f}  {p1000:9.4f}  "
          f"{'+' if p1000-p300>=0 else ''}{p1000-p300:.4f}")
    print("─" * 66)

    # ── interpretation ────────────────────────────────────────────────────
    print("\n[4] Interpretation")
    pos_gap = p1000 - p300
    if p1000 > 0.85:
        verdict = ("POSITION INFO IS PRESENT — the 300ep/1000-frame probe "
                   "was underfitting. embed_dim=32 is sufficient.")
    elif pos_gap > 0.15:
        verdict = ("PARTIAL RECOVERY — position info is in the embedding "
                   "but requires a stronger probe. More data/epochs help.")
    else:
        verdict = ("POSITION GENUINELY LOST — the 32-dim embedding has traded "
                   "absolute position for motion. embed_dim=64 may recover it.")
    print(f"  pos R² @300ep={p300:.3f}  →  @1000ep={p1000:.3f}  (Δ={pos_gap:+.3f})")
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()
