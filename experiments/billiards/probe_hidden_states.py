"""probe_hidden_states.py — Option B diagnostic: probe ViT hidden states vs projected embedding.

Extracts representations at TWO points in the FrameStack model's encoding path:

  input (9-ch) → ViT encoder → CLS token (192-dim)  ← pre-projector
                                       ↓
                             MLP projector → embedding (32-dim)  ← post-projector

Trains identical probes on both and compares position / velocity R².

If position R² is high (≈0.98) at 192-dim but low (≈0.59) at 32-dim:
  → eviction happens in the projector; wider embed_dim or aux loss on projector input will fix it.

If position R² is already low (≈0.59) at 192-dim:
  → the ViT encoder itself stopped encoding position; training objective must change.

Run from le-wm/:
    uv run python experiments/billiards/probe_hidden_states.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdf5plugin  # noqa: F401
import h5py
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # le-wm/

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

# ── probe parameters (same as extended probe for comparability) ────────────
PROBE_EPOCHS   = 1000
N_PROBE_FRAMES = 5000
PROBE_LR       = 1e-3
PROBE_BATCH    = 256
PROBE_HIDDEN   = 64
STATE_DIM      = 10
PROBE_SEED     = 42
LOG_EVERY      = 200


# ─────────────────────────────────────────────────────────────────────────────
# Representation extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_both_representations(model, f, frame_ep, frame_loc, ep_offset,
                                 n_frames: int, seed: int):
    """Return (hidden_192, projected_32, states) for n_frames random frames.

    hidden_192  : (N, 192)  — ViT CLS token, before the MLP projector
    projected_32: (N,  32)  — after the MLP projector  (same as encode())
    states      : (N,  10)  — ground-truth state vector
    """
    rng     = np.random.default_rng(seed)
    n_total = f["pixels"].shape[0]
    indices = np.sort(rng.choice(n_total, size=n_frames, replace=False))
    stacked = is_stacked_model(model)

    hidden_list, proj_list, state_list = [], [], []
    batch_size = 32

    for start in range(0, n_frames, batch_size):
        batch_idx = indices[start : start + batch_size]
        states_np = f["state"][batch_idx]

        # ── build image batch ────────────────────────────────────────────
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
            # (B, 9, H, W) — feed directly to encoder (no time dimension)
            imgs_flat = torch.stack(imgs).to(DEVICE)
        else:
            frames_np = f["pixels"][batch_idx]
            imgs_flat = torch.stack([preprocess(frames_np[j])
                                     for j in range(len(batch_idx))]).to(DEVICE)

        # ── pre-projector: ViT CLS token (192-dim) ───────────────────────
        enc_out   = model.encoder(imgs_flat, interpolate_pos_encoding=True)
        cls_token = enc_out.last_hidden_state[:, 0]          # (B, 192)

        # ── post-projector: MLP output (32-dim) ──────────────────────────
        projected = model.projector(cls_token)               # (B, 32)

        hidden_list.append(cls_token.cpu())
        proj_list.append(projected.cpu())
        state_list.append(torch.from_numpy(states_np.astype(np.float32)))

    return (
        torch.cat(hidden_list,  dim=0),   # (N, 192)
        torch.cat(proj_list,    dim=0),   # (N,  32)
        torch.cat(state_list,   dim=0),   # (N,  10)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probe
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


def train_probe(embs: torch.Tensor, states_n: torch.Tensor,
                label: str) -> np.ndarray:
    """Train probe; return final R² array (10 dims)."""
    dim   = embs.shape[-1]
    probe = StateProbe(dim).to(DEVICE)
    opt   = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    ed    = embs.to(DEVICE)
    sd    = states_n.to(DEVICE)
    n     = len(ed)

    print(f"\n  [{label}]  embed_dim={dim}  n={n}")
    for epoch in range(PROBE_EPOCHS + 1):
        if epoch > 0:
            perm = torch.randperm(n, device=DEVICE)
            for bi in range(0, n, PROBE_BATCH):
                idx  = perm[bi : bi + PROBE_BATCH]
                loss = F.mse_loss(probe(ed[idx]), sd[idx])
                opt.zero_grad(); loss.backward(); opt.step()

        if epoch % LOG_EVERY == 0:
            with torch.no_grad():
                p = probe(ed)
            ss_res = (p - sd).pow(2).sum(0)
            ss_tot = (sd - sd.mean(0)).pow(2).sum(0).clamp(min=1e-8)
            r2     = (1 - ss_res / ss_tot).cpu().numpy()
            print(f"    epoch {epoch:4d}/{PROBE_EPOCHS}"
                  f"  vel_R²={r2[VEL_DIMS].mean():.4f}"
                  f"  pos_R²(tgt)={r2[POS_DIMS].mean():.4f}")

    probe.eval()
    with torch.no_grad():
        p = probe(ed)
    ss_res = (p - sd).pow(2).sum(0)
    ss_tot = (sd - sd.mean(0)).pow(2).sum(0).clamp(min=1e-8)
    return (1 - ss_res / ss_tot).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Option B diagnostic — ViT hidden states vs projected embedding")
    print(f"  PROBE_EPOCHS={PROBE_EPOCHS}  N_PROBE_FRAMES={N_PROBE_FRAMES}")
    print("=" * 66)

    # ── dataset ──────────────────────────────────────────────────────────
    print(f"\n[0] Loading dataset: {DATASET_PATH.name}")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {f['pixels'].shape[0]:,} frames")

    print("    Building episode lookup …")
    frame_ep, frame_loc = build_episode_lookup(ep_len, ep_offset)

    # ── model ─────────────────────────────────────────────────────────────
    print("\n[1] Loading FrameStack checkpoint")
    model = load_model(CKPT_FRAMESTACKED, "FrameStack")
    model.eval().requires_grad_(False)

    # ── extract representations ───────────────────────────────────────────
    print(f"\n[2] Extracting {N_PROBE_FRAMES:,} frames at both levels …")
    hidden, projected, states = extract_both_representations(
        model, f, frame_ep, frame_loc, ep_offset, N_PROBE_FRAMES, seed=PROBE_SEED
    )
    print(f"    ViT CLS   (pre-proj) : {tuple(hidden.shape)}"
          f"  mean={hidden.mean():.3f}  std={hidden.std():.3f}")
    print(f"    Projected (post-proj): {tuple(projected.shape)}"
          f"  mean={projected.mean():.3f}  std={projected.std():.3f}")

    # normalise targets once, share across both probes
    state_mean = states.mean(0)
    state_std  = states.std(0).clamp(min=1e-6)
    states_n   = (states - state_mean) / state_std

    # ── probe 1: ViT CLS token (192-dim) ─────────────────────────────────
    print("\n[3] Probe A — ViT CLS token (192-dim, pre-projector)")
    r2_192 = train_probe(hidden, states_n, "192-dim pre-projector")

    # ── probe 2: projected embedding (32-dim) ────────────────────────────
    print("\n[4] Probe B — projected embedding (32-dim, post-projector)")
    r2_32  = train_probe(projected, states_n, "32-dim post-projector")

    # ── comparison table ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  {'Dimension':<12}  {'192-dim (ViT)':>14}  {'32-dim (proj)':>14}  {'Δ':>8}")
    print("─" * 70)
    for i, name in enumerate(STATE_NAMES):
        tag   = "[VEL]" if i in VEL_DIMS else "[POS]" if i in POS_DIMS else "     "
        delta = r2_32[i] - r2_192[i]
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<12}  {r2_192[i]:14.4f}  {r2_32[i]:14.4f}  {sign}{delta:.4f}  {tag}")
    print("─" * 70)

    v192 = r2_192[VEL_DIMS].mean()
    p192 = r2_192[POS_DIMS].mean()
    v32  = r2_32[VEL_DIMS].mean()
    p32  = r2_32[POS_DIMS].mean()
    print(f"  {'vel R² (mean)':<12}  {v192:14.4f}  {v32:14.4f}  "
          f"{'+' if v32-v192>=0 else ''}{v32-v192:.4f}")
    print(f"  {'pos R² (tgt)':<12}  {p192:14.4f}  {p32:14.4f}  "
          f"{'+' if p32-p192>=0 else ''}{p32-p192:.4f}")
    print("─" * 70)

    # ── interpretation ────────────────────────────────────────────────────
    print("\n[5] Interpretation")
    pos_drop = p192 - p32
    if p192 > 0.85:
        where  = "PROJECTOR"
        detail = (f"Position R²={p192:.3f} at 192-dim → drops to {p32:.3f} at 32-dim. "
                  f"The ViT encoder retains position information; the MLP projector "
                  f"evicts it when compressing to {projected.shape[-1]} dims. "
                  f"Fix: widen embed_dim, or add auxiliary loss on the 192-dim CLS token.")
    elif p192 > 0.70:
        where  = "BOTH (partial eviction in encoder)"
        detail = (f"Position R²={p192:.3f} at 192-dim — already below the baseline "
                  f"Transformer (0.983) but still meaningful. Further drop to {p32:.3f} "
                  f"at 32-dim. Both encoder and projector contribute to eviction.")
    else:
        where  = "ENCODER"
        detail = (f"Position R²={p192:.3f} at 192-dim — already lost before the projector. "
                  f"The ViT encoder itself stopped encoding position. "
                  f"Fix requires changing the training objective (aux loss or contrastive).")

    print(f"  Eviction location: {where}")
    print(f"  pos R²: 192-dim={p192:.3f}  →  32-dim={p32:.3f}  (drop={pos_drop:+.3f})")
    print(f"  → {detail}")


if __name__ == "__main__":
    main()
