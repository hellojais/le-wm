"""compare_models.py

Side-by-side diagnostic comparison:
  Model A (original) : lewm_epoch_8_object.ckpt    (embed_dim=192, sigreg=0.09)
  Model B (new)      : lewm_small_epoch_8_object.ckpt (embed_dim=32,  sigreg=0.01)

Three diagnostics:
  1. Pairwise embedding distances  (mean/std/min/max)
  2. Semantic geometry test        (near-pocket vs far-from-pocket frames)
  3. Single-step prediction accuracy

Run from the le-wm directory:
    uv run python compare_models.py
"""

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import hdf5plugin
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
from game import POCKET_POSITIONS, POCKET_RADIUS

CKPT_A       = Path.home() / ".stable_worldmodel" / "lewm_epoch_8_object.ckpt"
CKPT_B       = Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"
DEVICE       = torch.device("mps") if torch.backends.mps.is_available() \
               else torch.device("cpu")

_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

N_SAMPLE         = 50    # frames for pairwise distance check
N_NEAR           = 40    # near-pocket frames for semantic geometry
N_FAR            = 40    # far-from-pocket frames
NEAR_THRESH      = 80.0  # target ball within 80 px of a pocket = "near"
FAR_THRESH       = 200.0 # target ball beyond 200 px of all pockets = "far"
PRED_TEST_STEPS  = 10    # transitions for prediction accuracy


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame_hwc: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame_hwc.copy()).float() / 255.0
    return (t.permute(2, 0, 1) - _MEAN) / _STD


def encode_batch(model, frames_hwc: list) -> np.ndarray:
    """List of HWC uint8 frames → (N, D) float32 numpy array."""
    imgs = torch.stack([preprocess(f) for f in frames_hwc])  # (N, C, H, W)
    imgs = imgs.unsqueeze(0).to(DEVICE)                       # (1, N, C, H, W)
    with torch.inference_mode():
        emb = model.encode({"pixels": imgs})["emb"][0]        # (N, D)
    return emb.cpu().float().numpy()


def min_dist_to_pocket(target_xy: np.ndarray) -> float:
    return float(np.linalg.norm(POCKET_POSITIONS - target_xy, axis=1).min())


def load_model(path: Path, label: str):
    print(f"  Loading {label} : {path.name}")
    m = torch.load(path, map_location="cpu", weights_only=False)
    m = m.to(DEVICE).eval()
    m.requires_grad_(False)
    print(f"    {sum(p.numel() for p in m.parameters()):,} params  "
          f"embed_dim={m.projector.net[-1].out_features}")
    return m


def row(label, a_val, b_val, higher_is_better=False):
    """Print one comparison row, marking which model wins."""
    if isinstance(a_val, float) and isinstance(b_val, float):
        if higher_is_better:
            winner = "B ✓" if b_val > a_val else ("A ✓" if a_val > b_val else "tie")
        else:
            winner = "B ✓" if b_val < a_val else ("A ✓" if a_val < b_val else "tie")
        print(f"  {label:<38s}  {a_val:>10.4f}  {b_val:>10.4f}  {winner}")
    else:
        print(f"  {label:<38s}  {str(a_val):>10s}  {str(b_val):>10s}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  LeWM Billiards — Model Comparison")
print("=" * 70)
print()
print("Loading models …")
model_a = load_model(CKPT_A, "Model A (original)")
model_b = load_model(CKPT_B, "Model B (new small)")

print("\nLoading dataset …")
f         = h5py.File(DATASET_PATH, "r", swmr=True)
ep_len    = f["ep_len"][:]
ep_offset = f["ep_offset"][:]
states    = f["state"]     # shape (N, 10): [cx,cy,cvx,cvy, tx,ty,tvx,tvy, px,py]
pixels    = f["pixels"]
actions   = f["action"]
n_frames  = pixels.shape[0]

acts      = actions[:]
act_mean  = acts.mean(0).astype(np.float32)
act_std   = acts.std(0).astype(np.float32)
act_std   = np.where(act_std < 1e-6, 1.0, act_std)

print(f"  ✓ {len(ep_len)} episodes, {n_frames:,} frames")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1 — PAIRWISE EMBEDDING DISTANCES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  DIAGNOSTIC 1 — Pairwise embedding distances")
print(f"  ({N_SAMPLE} randomly sampled frames, all {N_SAMPLE*(N_SAMPLE-1)//2} pairs)")
print("─" * 70)

rng = np.random.default_rng(42)
idx = rng.choice(n_frames, size=N_SAMPLE, replace=False)
frames_sample = [pixels[int(i)] for i in idx]

embs_a = encode_batch(model_a, frames_sample)   # (N, 192)
embs_b = encode_batch(model_b, frames_sample)   # (N, 32)

def pairwise_l2(embs):
    dists = []
    for i, j in combinations(range(len(embs)), 2):
        dists.append(np.linalg.norm(embs[i] - embs[j]))
    return np.array(dists)

dists_a = pairwise_l2(embs_a)
dists_b = pairwise_l2(embs_b)

print(f"\n  {'Metric':<38s}  {'Model A':>10s}  {'Model B':>10s}  Winner")
print(f"  {'':38s}  {'(orig)':>10s}  {'(new)':>10s}")
print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*6}")
row("Mean pairwise L2",        dists_a.mean(), dists_b.mean())
row("Std  pairwise L2",        dists_a.std(),  dists_b.std(),  higher_is_better=True)
row("Min  pairwise L2",        dists_a.min(),  dists_b.min())
row("Max  pairwise L2",        dists_a.max(),  dists_b.max())
row("Coeff of variation (std/mean)",
    dists_a.std()/dists_a.mean(),
    dists_b.std()/dists_b.mean(),
    higher_is_better=True)

print(f"\n  Interpretation:")
print(f"    Original model: mean={dists_a.mean():.2f}, std={dists_a.std():.2f}  "
      f"→ CV={dists_a.std()/dists_a.mean():.3f}")
print(f"    New model     : mean={dists_b.mean():.2f}, std={dists_b.std():.2f}  "
      f"→ CV={dists_b.std()/dists_b.mean():.3f}")
print(f"    (Higher CV = more variation = less uniform = more structured)")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2 — SEMANTIC GEOMETRY (near-pocket vs far-from-pocket)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  DIAGNOSTIC 2 — Semantic geometry test")
print(f"  Near-pocket frames (dist < {NEAR_THRESH} px) vs")
print(f"  Far-from-pocket frames (dist > {FAR_THRESH} px)")
print("─" * 70)

# Compute per-frame min dist to pocket using the state array
# state[i] = [cx, cy, cvx, cvy, tx, ty, tvx, tvy, px, py]
# tx, ty = target ball position = state[i, 4:6]
print("\n  Scanning dataset for near/far frames …")
all_states = states[:]   # load fully for speed
target_xy  = all_states[:, 4:6]   # (N, 2)
pocket_dists = np.array([
    np.linalg.norm(POCKET_POSITIONS - target_xy[i], axis=1).min()
    for i in range(0, n_frames, 100)   # sample every 100th frame for speed
])
frame_idx_sample = np.arange(0, n_frames, 100)

near_mask = pocket_dists < NEAR_THRESH
far_mask  = pocket_dists > FAR_THRESH
print(f"  Found {near_mask.sum()} near-pocket frames / "
      f"{far_mask.sum()} far-from-pocket frames (in sampled set)")

# Pick N_NEAR / N_FAR random frames from each group
rng2 = np.random.default_rng(99)
near_idx = rng2.choice(frame_idx_sample[near_mask], size=min(N_NEAR, near_mask.sum()), replace=False)
far_idx  = rng2.choice(frame_idx_sample[far_mask],  size=min(N_FAR,  far_mask.sum()),  replace=False)

near_frames = [pixels[int(i)] for i in near_idx]
far_frames  = [pixels[int(i)] for i in far_idx]
all_geo_frames = near_frames + far_frames
labels_geo = ["near"] * len(near_frames) + ["far"] * len(far_frames)

embs_geo_a = encode_batch(model_a, all_geo_frames)
embs_geo_b = encode_batch(model_b, all_geo_frames)

n_near = len(near_frames)
n_far  = len(far_frames)

def within_between_dists(embs, n_near, n_far):
    """
    Returns:
        within_near  : mean L2 among near-pocket embeddings
        within_far   : mean L2 among far-from-pocket embeddings
        between      : mean L2 between near and far embeddings
    """
    near_embs = embs[:n_near]
    far_embs  = embs[n_near:]

    wn = np.mean([np.linalg.norm(near_embs[i] - near_embs[j])
                  for i, j in combinations(range(n_near), 2)])
    wf = np.mean([np.linalg.norm(far_embs[i] - far_embs[j])
                  for i, j in combinations(range(n_far), 2)])
    bt = np.mean([np.linalg.norm(near_embs[i] - far_embs[j])
                  for i in range(n_near) for j in range(n_far)])
    return wn, wf, bt

wn_a, wf_a, bt_a = within_between_dists(embs_geo_a, n_near, n_far)
wn_b, wf_b, bt_b = within_between_dists(embs_geo_b, n_near, n_far)

# Separation score: between / within (higher = more semantically structured)
sep_a = bt_a / ((wn_a + wf_a) / 2)
sep_b = bt_b / ((wn_b + wf_b) / 2)

print(f"\n  {'Metric':<38s}  {'Model A':>10s}  {'Model B':>10s}  Winner")
print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*6}")
row("Within-near L2 (near frames)",  wn_a, wn_b)
row("Within-far  L2 (far frames)",   wf_a, wf_b)
row("Between-group L2",              bt_a, bt_b)
row("Separation score (between/within)",  sep_a, sep_b, higher_is_better=True)

print(f"\n  Interpretation:")
print(f"    Separation > 1.0 means near-pocket frames are MORE different from")
print(f"    far-from-pocket frames than they are from each other.")
print(f"    Model A sep={sep_a:.3f}   Model B sep={sep_b:.3f}")
if sep_b > sep_a:
    print(f"    → New model has {sep_b/sep_a:.2f}× better semantic separation ✓")
else:
    print(f"    → Original model has better semantic separation")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 3 — SINGLE-STEP PREDICTION ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  DIAGNOSTIC 3 — Single-step prediction accuracy")
print(f"  ({PRED_TEST_STEPS} transitions from episode 500)")
print("─" * 70)

ep    = 500
off   = int(ep_offset[ep])

print(f"\n  {'t':>6}  {'A pred→real L2':>16}  {'B pred→real L2':>16}  "
      f"{'A start→real L2':>16}")

mse_a_list, mse_b_list = [], []

with torch.inference_mode():
    for t in range(PRED_TEST_STEPS):
        frame_t  = pixels[off + t]
        frame_t1 = pixels[off + t + 1]
        act_t    = actions[off + t]
        act_norm = ((act_t - act_mean) / act_std).astype(np.float32)

        def predict_one(model):
            img_t  = preprocess(frame_t).unsqueeze(0).unsqueeze(0).to(DEVICE)
            img_t1 = preprocess(frame_t1).unsqueeze(0).unsqueeze(0).to(DEVICE)
            emb_t  = model.encode({"pixels": img_t})["emb"]         # (1,1,D)
            emb_t1 = model.encode({"pixels": img_t1})["emb"][0, 0]  # (D,)
            act_e  = model.action_encoder(
                torch.from_numpy(act_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            )                                                         # (1,1,A)
            pred   = model.predict(emb_t, act_e)[0, -1]             # (D,)
            l2     = (pred - emb_t1).norm().item()
            mse    = F.mse_loss(pred, emb_t1).item()
            l2_0   = (emb_t[0, 0] - emb_t1).norm().item()
            return l2, mse, l2_0

        l2_a, mse_a, l2_0_a = predict_one(model_a)
        l2_b, mse_b, _      = predict_one(model_b)
        mse_a_list.append(mse_a)
        mse_b_list.append(mse_b)

        winner = "B" if l2_b < l2_a else "A"
        print(f"  t={off+t:>6d}  {l2_a:>16.4f}  {l2_b:>16.4f}  "
              f"{l2_0_a:>16.4f}  [{winner}]")

print(f"\n  {'Metric':<38s}  {'Model A':>10s}  {'Model B':>10s}  Winner")
print(f"  {'-'*38}  {'-'*10}  {'-'*10}  {'-'*6}")
row("Mean pred→real L2",
    float(np.mean([np.sqrt(m * embs_geo_a.shape[1]) for m in mse_a_list])),
    float(np.mean([np.sqrt(m * embs_geo_b.shape[1]) for m in mse_b_list])))
row("Mean pred MSE (val/pred_loss metric)",
    float(np.mean(mse_a_list)), float(np.mean(mse_b_list)))


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY COMPARISON")
print("=" * 70)
print(f"  {'Metric':<38s}  {'Model A':>10s}  {'Model B':>10s}")
print(f"  {'':38s}  {'orig/192d':>10s}  {'new/32d':>10s}")
print(f"  {'-'*38}  {'-'*10}  {'-'*10}")
print(f"  {'embed_dim':<38s}  {'192':>10s}  {'32':>10s}")
print(f"  {'sigreg weight':<38s}  {'0.09':>10s}  {'0.01':>10s}")
print(f"  {'training epochs':<38s}  {'20':>10s}  {'10':>10s}")
print(f"  {'val/pred_loss (checkpoint)':<38s}  {'0.00737':>10s}  {'0.00284':>10s}")
row("Pairwise L2 mean",            dists_a.mean(),  dists_b.mean())
row("Pairwise L2 std",             dists_a.std(),   dists_b.std(),  higher_is_better=True)
row("CV (std/mean)",               dists_a.std()/dists_a.mean(), dists_b.std()/dists_b.mean(), higher_is_better=True)
row("Semantic separation score",   sep_a, sep_b, higher_is_better=True)
row("Mean pred MSE",               float(np.mean(mse_a_list)), float(np.mean(mse_b_list)))
print("=" * 70)

f.close()
print("\nDone.")
