"""diagnose_billiards.py

Four diagnostic checks for the LeWM billiards evaluation:
  1. Embedding sanity check
  2. Action range check
  3. Goal distance sanity check (pre-planning difficulty)
  4. Single-step prediction check (model accuracy)

Run from the le-wm directory:
    uv run python diagnose_billiards.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import hdf5plugin  # must be before h5py
import h5py
from itertools import combinations

# ── paths & device ───────────────────────────────────────────────────────────
CHECKPOINT   = Path.home() / ".stable_worldmodel" / "lewm_epoch_19_object.ckpt"
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"
DEVICE       = torch.device("mps") if torch.backends.mps.is_available() \
               else torch.device("cpu")
HISTORY_SIZE = 3

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

# ── helpers ───────────────────────────────────────────────────────────────────
def preprocess(frame_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW float32 ImageNet-normalised."""
    t = torch.from_numpy(frame_hwc.copy()).float() / 255.0
    t = t.permute(2, 0, 1)
    return (t - _MEAN) / _STD

def encode_frames(model, frames_hwc):
    """List of HWC ndarrays → (N, D) embedding tensor."""
    imgs = torch.stack([preprocess(f) for f in frames_hwc])  # (N, C, H, W)
    imgs = imgs.unsqueeze(0).to(DEVICE)                       # (1, N, C, H, W)
    with torch.inference_mode():
        out = model.encode({"pixels": imgs})
    return out["emb"][0]  # (N, D)

def sep(title=""):
    width = 60
    print("\n" + "─" * width)
    if title:
        print(f"  {title}")
        print("─" * width)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model …")
model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = model.to(DEVICE).eval()
model.requires_grad_(False)
print(f"  ✓ {sum(p.numel() for p in model.parameters()):,} params  device={DEVICE}")

print("Loading dataset …")
f         = h5py.File(DATASET_PATH, "r", swmr=True)
ep_len    = f["ep_len"][:]
ep_offset = f["ep_offset"][:]
n_frames  = f["pixels"].shape[0]
print(f"  ✓ {len(ep_len)} episodes, {n_frames:,} frames")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1 — EMBEDDING SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
sep("DIAGNOSTIC 1 — Embedding sanity check")

rng = np.random.default_rng(42)
sample_indices = rng.choice(n_frames, size=10, replace=False)
sample_frames  = [f["pixels"][int(i)] for i in sample_indices]

embs = encode_frames(model, sample_frames)  # (10, D)
embs_np = embs.cpu().float().numpy()

print(f"\n  Embedding shape : {embs_np.shape}")
print(f"  Per-dim mean    : mean={embs_np.mean():.4f}  std={embs_np.std():.4f}")

per_dim_std = embs_np.std(axis=0)
print(f"\n  Per-dimension std across 10 frames:")
print(f"    min={per_dim_std.min():.4f}  max={per_dim_std.max():.4f}  "
      f"mean={per_dim_std.mean():.4f}  median={np.median(per_dim_std):.4f}")
print(f"    dims with std < 0.01 : {(per_dim_std < 0.01).sum()}  "
      f"(dead / unexpressive dims)")
print(f"    dims with std > 1.0  : {(per_dim_std > 1.0).sum()}")

print(f"\n  Pairwise L2 distances between all 10 embeddings:")
dists = []
for (i, ei), (j, ej) in combinations(enumerate(embs_np), 2):
    d = np.linalg.norm(ei - ej)
    dists.append(d)
    print(f"    frames {sample_indices[i]:6d} ↔ {sample_indices[j]:6d} : {d:.4f}")
dists = np.array(dists)
print(f"\n  Summary: min={dists.min():.4f}  max={dists.max():.4f}  "
      f"mean={dists.mean():.4f}  std={dists.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2 — ACTION RANGE CHECK
# ─────────────────────────────────────────────────────────────────────────────
sep("DIAGNOSTIC 2 — Action range check")

acts = f["action"][:]   # (N, 2)
print(f"\n  Dataset actions  (shape {acts.shape}):")
print(f"    min  = {acts.min(axis=0)}")
print(f"    max  = {acts.max(axis=0)}")
print(f"    mean = {acts.mean(axis=0)}")
print(f"    std  = {acts.std(axis=0)}")

act_mean = acts.mean(axis=0).astype(np.float32)
act_std  = acts.std(axis=0).astype(np.float32)
act_std  = np.where(act_std < 1e-6, 1.0, act_std)

print(f"\n  CEM samples in NORMALISED space:  clamp(-3, 3)")
print(f"  Denormalised to env units via:    action = norm * std + mean")
cem_lo = -3.0 * act_std + act_mean
cem_hi =  3.0 * act_std + act_mean
print(f"    CEM env-unit range dim0: [{cem_lo[0]:.3f}, {cem_hi[0]:.3f}]")
print(f"    CEM env-unit range dim1: [{cem_lo[1]:.3f}, {cem_hi[1]:.3f}]")

print(f"\n  BilliardsEnv action_space: low=-10, high=+10")
print(f"  Dataset action range      : [{acts.min():.3f}, {acts.max():.3f}]")
print(f"  CEM samples reach         : [{cem_lo.min():.3f}, {cem_hi.max():.3f}]")
coverage = (cem_hi.max() - cem_lo.min()) / (10.0 - (-10.0)) * 100
print(f"  Coverage of env range     : {coverage:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 3 — GOAL DISTANCE BEFORE PLANNING
# ─────────────────────────────────────────────────────────────────────────────
sep("DIAGNOSTIC 3 — Goal distance (pre-planning difficulty)")

combos = [
    ("Combo A (same ep)", 500, 0, 500),
    ("Combo B (cross-ep)", 100, 0, 3000),
]

for label, start_ep, start_fi, goal_ep in combos:
    s_off = ep_offset[start_ep]
    g_off = ep_offset[goal_ep]
    g_len = ep_len[goal_ep]

    start_frame = f["pixels"][s_off + start_fi]
    goal_frame  = f["pixels"][g_off + g_len - 4]  # same offset as evaluate_billiards.py

    emb_start = encode_frames(model, [start_frame])[0]  # (D,)
    emb_goal  = encode_frames(model, [goal_frame])[0]   # (D,)

    dist_l2  = (emb_start - emb_goal).norm().item()
    dist_mse = F.mse_loss(emb_start, emb_goal, reduction="sum").item()

    print(f"\n  {label}")
    print(f"    start frame  : ep {start_ep} frame {start_fi}")
    print(f"    goal  frame  : ep {goal_ep} frame {g_len - 4}")
    print(f"    L2   distance: {dist_l2:.4f}")
    print(f"    MSE·D (cost) : {dist_mse:.4f}   ← this is what CEM minimises")

    # Also encode a few consecutive frames around start to see typical distances
    nearby_frames = [f["pixels"][s_off + max(0, start_fi + dt)] for dt in range(5)]
    nearby_embs   = encode_frames(model, nearby_frames)  # (5, D)
    step_dists = [(nearby_embs[i] - nearby_embs[i+1]).norm().item()
                  for i in range(len(nearby_embs) - 1)]
    print(f"    Typical 1-step distance in this ep: "
          f"mean={np.mean(step_dists):.4f}  "
          f"({', '.join(f'{d:.4f}' for d in step_dists)})")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 4 — SINGLE-STEP PREDICTION CHECK
# ─────────────────────────────────────────────────────────────────────────────
sep("DIAGNOSTIC 4 — Single-step prediction accuracy")

ep = 500
off = int(ep_offset[ep])

# Test on 5 consecutive transitions within ep 500
print(f"\n  Episode {ep}, testing 5 consecutive transitions:")
print(f"  {'t':>4}  {'pred→real L2':>14}  {'pred→real MSE':>14}  "
      f"{'start→real L2':>14}")

errors_mse = []
with torch.inference_mode():
    for t in range(5):
        # Frames t and t+1
        frame_t   = f["pixels"][off + t]
        frame_t1  = f["pixels"][off + t + 1]
        action_t  = f["action"][off + t]

        # Encode frame t as a 1-frame "history"
        img_t  = preprocess(frame_t).unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1,1,C,H,W)
        info_t = model.encode({"pixels": img_t})
        emb_t  = info_t["emb"]  # (1, 1, D)

        # Action embedding
        act_norm = torch.from_numpy(
            ((action_t - act_mean) / act_std).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 2)
        act_emb = model.action_encoder(act_norm)  # (1, 1, A_emb)

        # Predict next embedding
        pred_emb = model.predict(emb_t, act_emb)[:, -1, :]  # (1, D)

        # Real next embedding
        img_t1    = preprocess(frame_t1).unsqueeze(0).unsqueeze(0).to(DEVICE)
        info_t1   = model.encode({"pixels": img_t1})
        emb_t1    = info_t1["emb"][0, 0]  # (D,)
        pred_emb_ = pred_emb[0]           # (D,)

        mse  = F.mse_loss(pred_emb_, emb_t1, reduction="mean").item()
        l2   = (pred_emb_ - emb_t1).norm().item()
        l2_0 = (emb_t[0, 0] - emb_t1).norm().item()

        errors_mse.append(mse)
        print(f"  t={t+off:>6d}  {l2:>14.6f}  {mse:>14.6f}  {l2_0:>14.6f}")

print(f"\n  Mean MSE over 5 steps: {np.mean(errors_mse):.6f}")
print(f"  (training val/pred_loss was 0.00737 — same metric?)")
print(f"\n  Note: val/pred_loss may be computed differently (e.g. on projector")
print(f"  output space). The above uses the raw encoder embedding space.")

f.close()
print("\n" + "─" * 60)
print("  Diagnostics complete.")
print("─" * 60)
