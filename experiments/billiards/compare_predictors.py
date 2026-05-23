"""compare_predictors.py

Three-way comparison: Transformer vs Mamba vs Frame Stacking predictors.

Experiments
───────────
  Exp 1 – Transformer  : baseline ARPredictor, 3-channel input
  Exp 2 – Mamba        : S6 SSM predictor,     3-channel input (same encoder)
  Exp 3 – Frame Stack  : Transformer predictor, 9-channel input (3 frames stacked)

Checkpoints
───────────
  Transformer : ~/.stable_worldmodel/lewm_small_epoch_8_object.ckpt
  Mamba       : ~/.stable_worldmodel/lewm_mamba_best_object.ckpt
  FrameStack  : ~/.stable_worldmodel/lewm_framestacked_best_object.ckpt

Encoding notes
──────────────
  Transformer / Mamba : model.encode({"pixels": (B, T, 3, 96, 96)})
  Frame Stacking      : model.encode({"pixels": (B, T, 9, 96, 96)})
                        where each of the 9 channels = [frame_t-2, frame_t-1, frame_t]
                        (3 frames × 3 RGB channels).  For pairwise / probe metrics,
                        episode-boundary clamping is applied: if t < 2, frame 0 is
                        repeated (same padding as used during training).

Metrics
───────
  val/pred_loss      — mean single-step prediction MSE over N_PRED_STEPS transitions
  velocity R²        — probe R² mean over (cue_vx, cue_vy, tgt_vx, tgt_vy)
  position R²        — probe R² mean over (tgt_x, tgt_y)
  pairwise L2 mean/σ — distribution of inter-embedding distances

Output
───────
  Prints a three-column table to stdout.
  Saves comparison plot to:  results/three_way_comparison.png

Run from le-wm/:
    uv run python experiments/billiards/compare_predictors.py
"""

import sys
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdf5plugin  # noqa: F401 — must import before h5py
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── ensure module_mamba is importable before any checkpoint is unpickled ──
sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # le-wm/

# ── paths ─────────────────────────────────────────────────────────────────
CKPT_TRANSFORMER = (
    Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
)
CKPT_MAMBA = (
    Path.home() / ".stable_worldmodel" / "lewm_mamba_best_object.ckpt"
)
CKPT_FRAMESTACKED = (
    Path.home() / ".stable_worldmodel" / "lewm_framestacked_best_object.ckpt"
)
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"
OUT_DIR      = Path(__file__).parent.parent.parent / "results"
OUT_PNG      = OUT_DIR / "three_way_comparison.png"

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ── diagnostic parameters ─────────────────────────────────────────────────
N_PAIR_FRAMES  = 20     # frames for pairwise distance stats
N_PRED_STEPS   = 20     # transitions for prediction MSE
N_PROBE_FRAMES = 1000   # frames for probe training
PROBE_EPOCHS   = 300
PROBE_LR       = 1e-3
PROBE_BATCH    = 128
PROBE_HIDDEN   = 64
STATE_DIM      = 10
PROBE_SEED     = 42

STATE_NAMES = [
    "cue_x", "cue_y", "cue_vx", "cue_vy",
    "tgt_x", "tgt_y", "tgt_vx", "tgt_vy",
    "pkt_x", "pkt_y",
]
VEL_DIMS  = [2, 3, 6, 7]    # cue_vx, cue_vy, tgt_vx, tgt_vy
POS_DIMS  = [4, 5]           # tgt_x, tgt_y  (planning-critical)

# ImageNet normalisation
_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW float32 ImageNet-normalised (3, H, W)."""
    t = torch.from_numpy(frame_hwc.copy()).float() / 255.0
    return (t.permute(2, 0, 1) - _MEAN) / _STD


def load_model(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {label}:\n  {path}\n"
            "Train the model first, then re-run this script."
        )
    print(f"  Loading {label:14s} ← {path.name}")
    m = torch.load(path, map_location="cpu", weights_only=False)
    m = m.to(DEVICE).eval()
    m.requires_grad_(False)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"    {n_params:,} params   device={DEVICE}")
    return m


def get_embed_dim(model) -> int:
    """Infer embed_dim by running a dummy forward pass."""
    c = model.encoder.config.num_channels   # 3 or 9
    with torch.no_grad():
        dummy = torch.zeros(1, 1, c, 96, 96, device=DEVICE)
        emb = model.encode({"pixels": dummy})["emb"]
    return emb.shape[-1]


def is_stacked_model(model) -> bool:
    """True if the model's ViT encoder expects 9-channel input."""
    return model.encoder.config.num_channels == 9


# ── Episode index helpers for frame-triplet construction ─────────────────

def build_episode_lookup(ep_len: np.ndarray, ep_offset: np.ndarray):
    """Build a flat array mapping each global frame index → (ep_idx, local_pos)."""
    n_total = int(ep_offset[-1]) + int(ep_len[-1])
    frame_ep  = np.empty(n_total, dtype=np.int32)
    frame_loc = np.empty(n_total, dtype=np.int32)
    for ep, (off, length) in enumerate(zip(ep_offset, ep_len)):
        frame_ep [off : off + length] = ep
        frame_loc[off : off + length] = np.arange(length)
    return frame_ep, frame_loc


def triplet_indices(global_idx: int, ep_off: int, local_pos: int) -> tuple:
    """Return global indices (g_t2, g_t1, g_t) for a frame-stack triplet."""
    g_t  = global_idx
    g_t1 = ep_off + max(0, local_pos - 1)
    g_t2 = ep_off + max(0, local_pos - 2)
    return g_t2, g_t1, g_t


def encode_frames(model, f: h5py.File, indices: np.ndarray,
                  frame_ep: np.ndarray, frame_loc: np.ndarray,
                  ep_offset: np.ndarray) -> np.ndarray:
    """Encode a list of global frame indices → (N, D) float32 numpy.

    For standard 3-channel models, encodes each frame independently.
    For 9-channel frame-stacking models, builds [t-2, t-1, t] triplets.
    """
    stacked = is_stacked_model(model)
    imgs_list = []

    for g in indices:
        g = int(g)
        if stacked:
            ep  = int(frame_ep[g])
            loc = int(frame_loc[g])
            off = int(ep_offset[ep])
            g2, g1, g0 = triplet_indices(g, off, loc)
            p2 = preprocess(f["pixels"][g2])
            p1 = preprocess(f["pixels"][g1])
            p0 = preprocess(f["pixels"][g0])
            imgs_list.append(torch.cat([p2, p1, p0], dim=0))  # (9, H, W)
        else:
            imgs_list.append(preprocess(f["pixels"][g]))       # (3, H, W)

    imgs = torch.stack(imgs_list).unsqueeze(0).to(DEVICE)     # (1, N, C, H, W)
    with torch.inference_mode():
        emb = model.encode({"pixels": imgs})["emb"][0]         # (N, D)
    return emb.cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Pairwise L2 distances
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_stats(model, f, frame_ep, frame_loc, ep_offset, n_frames: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(f["pixels"].shape[0], size=n_frames, replace=False)
    embs = encode_frames(model, f, idx, frame_ep, frame_loc, ep_offset)
    dists = [
        np.linalg.norm(embs[i] - embs[j])
        for (i, j) in combinations(range(len(embs)), 2)
    ]
    return float(np.mean(dists)), float(np.std(dists))


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Single-step prediction MSE
# ─────────────────────────────────────────────────────────────────────────────

def prediction_mse(model, f, ep_len, ep_offset, frame_ep, frame_loc,
                   n_steps: int, seed: int = 0):
    """Average single-step prediction MSE over n_steps random transitions."""
    stacked = is_stacked_model(model)
    acts = f["action"][:]
    act_mean = acts.mean(0).astype(np.float32)
    act_std  = acts.std(0).astype(np.float32)
    act_std  = np.where(act_std < 1e-6, 1.0, act_std)

    rng     = np.random.default_rng(seed)
    ep_idx  = rng.choice(len(ep_len), size=n_steps, replace=True)
    mse_all = []

    def _encode_single(global_idx):
        g = int(global_idx)
        if stacked:
            ep  = int(frame_ep[g])
            loc = int(frame_loc[g])
            off = int(ep_offset[ep])
            g2, g1, g0 = triplet_indices(g, off, loc)
            p2, p1, p0 = (preprocess(f["pixels"][g2]),
                          preprocess(f["pixels"][g1]),
                          preprocess(f["pixels"][g0]))
            img = torch.cat([p2, p1, p0], dim=0)              # (9, H, W)
        else:
            img = preprocess(f["pixels"][g])                   # (3, H, W)
        return img.unsqueeze(0).unsqueeze(0).to(DEVICE)        # (1, 1, C, H, W)

    with torch.inference_mode():
        for ep in ep_idx:
            off    = int(ep_offset[ep])
            length = int(ep_len[ep])
            if length < 2:
                continue
            t = int(rng.integers(0, length - 1))

            img_t  = _encode_single(off + t)
            img_t1 = _encode_single(off + t + 1)

            emb_t  = model.encode({"pixels": img_t })["emb"]   # (1,1,D)
            emb_t1 = model.encode({"pixels": img_t1})["emb"][0, 0]  # (D,)

            action_t = f["action"][off + t]
            act_norm = torch.from_numpy(
                ((action_t - act_mean) / act_std).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)             # (1,1,2)
            act_emb  = model.action_encoder(act_norm)           # (1,1,A)

            pred_emb = model.predict(emb_t, act_emb)[:, -1, :]  # (1,D)
            mse = F.mse_loss(pred_emb[0], emb_t1, reduction="mean").item()
            mse_all.append(mse)

    return float(np.mean(mse_all)) if mse_all else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Probe R² per state dimension
# ─────────────────────────────────────────────────────────────────────────────

class _StateProbe(nn.Module):
    def __init__(self, embed_dim, state_dim=STATE_DIM, hidden=PROBE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, state_dim)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _extract_probe_data(model, f, frame_ep, frame_loc, ep_offset, n_frames, seed):
    rng     = np.random.default_rng(seed)
    n_total = f["pixels"].shape[0]
    indices = np.sort(rng.choice(n_total, size=n_frames, replace=False))
    stacked = is_stacked_model(model)

    emb_list, state_list = [], []
    for start in range(0, n_frames, 32):
        batch_idx = indices[start : start + 32]
        states_np = f["state"][batch_idx]

        if stacked:
            imgs = []
            for g in batch_idx:
                g = int(g)
                ep  = int(frame_ep[g])
                loc = int(frame_loc[g])
                off = int(ep_offset[ep])
                g2, g1, g0 = triplet_indices(g, off, loc)
                p2, p1, p0 = (preprocess(f["pixels"][g2]),
                              preprocess(f["pixels"][g1]),
                              preprocess(f["pixels"][g0]))
                imgs.append(torch.cat([p2, p1, p0], dim=0))   # (9, H, W)
            imgs_t = torch.stack(imgs).unsqueeze(0).to(DEVICE) # (1, B, 9, H, W)
        else:
            frames_np = f["pixels"][batch_idx]
            imgs_t = torch.stack([preprocess(frames_np[j])
                                  for j in range(len(batch_idx))])
            imgs_t = imgs_t.unsqueeze(0).to(DEVICE)            # (1, B, 3, H, W)

        emb = model.encode({"pixels": imgs_t})["emb"][:, :, :][:, :len(batch_idx), :]
        emb = emb[0]                                            # (B, D)
        emb_list.append(emb.cpu())
        state_list.append(torch.from_numpy(states_np.astype(np.float32)))

    embs   = torch.cat(emb_list,   dim=0)
    states = torch.cat(state_list, dim=0)
    return embs, states


def probe_r2(model, f, frame_ep, frame_loc, ep_offset, embed_dim: int,
             seed: int = PROBE_SEED, label: str = ""):
    """Train a lightweight MLP probe; return R² per state dimension."""
    print(f"    Training probe (embed_dim={embed_dim}) …", flush=True)
    embs, states = _extract_probe_data(
        model, f, frame_ep, frame_loc, ep_offset, N_PROBE_FRAMES, seed
    )

    state_mean = states.mean(0)
    state_std  = states.std(0).clamp(min=1e-6)
    states_n   = (states - state_mean) / state_std

    probe    = _StateProbe(embed_dim).to(DEVICE)
    opt      = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    embs_d   = embs.to(DEVICE)
    states_d = states_n.to(DEVICE)
    n        = len(embs_d)

    for epoch in range(PROBE_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for bi in range(0, n, PROBE_BATCH):
            idx  = perm[bi : bi + PROBE_BATCH]
            loss = F.mse_loss(probe(embs_d[idx]), states_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                p = probe(embs_d)
            ss_res = (p - states_d).pow(2).sum(0)
            ss_tot = (states_d - states_d.mean(0)).pow(2).sum(0).clamp(min=1e-8)
            r2 = (1 - ss_res / ss_tot).cpu().numpy()
            print(f"      epoch {epoch:3d}/{PROBE_EPOCHS}  mean_R²={r2.mean():.4f}")

    probe.eval()
    with torch.no_grad():
        p = probe(embs_d)
    ss_res = (p - states_d).pow(2).sum(0)
    ss_tot = (states_d - states_d.mean(0)).pow(2).sum(0).clamp(min=1e-8)
    return (1 - ss_res / ss_tot).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  compare_predictors.py — Transformer vs Mamba vs Frame Stacking")
    print("=" * 72)

    # ── Load dataset ──────────────────────────────────────────────────
    print(f"\n[0] Loading dataset: {DATASET_PATH.name}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {len(ep_len)} episodes, {f['pixels'].shape[0]:,} frames")

    print("    Building episode lookup table …")
    frame_ep, frame_loc = build_episode_lookup(ep_len, ep_offset)
    print(f"    ✓ lookup table built ({len(frame_ep):,} entries)")

    # ── Load models ───────────────────────────────────────────────────
    print("\n[1] Loading checkpoints")
    model_tr = load_model(CKPT_TRANSFORMER, "Transformer")
    model_mb = load_model(CKPT_MAMBA,       "Mamba")
    model_fs = load_model(CKPT_FRAMESTACKED,"FrameStack")

    embed_tr = get_embed_dim(model_tr)
    embed_mb = get_embed_dim(model_mb)
    embed_fs = get_embed_dim(model_fs)
    print(f"    embed_dim: Transformer={embed_tr}  Mamba={embed_mb}  FrameStack={embed_fs}")
    print(f"    in_chans:  Transformer={model_tr.encoder.config.num_channels}"
          f"  Mamba={model_mb.encoder.config.num_channels}"
          f"  FrameStack={model_fs.encoder.config.num_channels}")

    # ── Metric 1 — Pairwise distances ─────────────────────────────────
    print("\n[2] Pairwise L2 distances")
    pw_mean_tr, pw_std_tr = pairwise_stats(
        model_tr, f, frame_ep, frame_loc, ep_offset, N_PAIR_FRAMES, seed=1)
    pw_mean_mb, pw_std_mb = pairwise_stats(
        model_mb, f, frame_ep, frame_loc, ep_offset, N_PAIR_FRAMES, seed=1)
    pw_mean_fs, pw_std_fs = pairwise_stats(
        model_fs, f, frame_ep, frame_loc, ep_offset, N_PAIR_FRAMES, seed=1)
    print(f"    Transformer: mean={pw_mean_tr:.4f}  std={pw_std_tr:.4f}")
    print(f"    Mamba      : mean={pw_mean_mb:.4f}  std={pw_std_mb:.4f}")
    print(f"    FrameStack : mean={pw_mean_fs:.4f}  std={pw_std_fs:.4f}")

    # ── Metric 2 — Prediction MSE ──────────────────────────────────────
    print("\n[3] Single-step prediction MSE")
    mse_tr = prediction_mse(
        model_tr, f, ep_len, ep_offset, frame_ep, frame_loc, N_PRED_STEPS, seed=2)
    mse_mb = prediction_mse(
        model_mb, f, ep_len, ep_offset, frame_ep, frame_loc, N_PRED_STEPS, seed=2)
    mse_fs = prediction_mse(
        model_fs, f, ep_len, ep_offset, frame_ep, frame_loc, N_PRED_STEPS, seed=2)
    print(f"    Transformer: {mse_tr:.6f}")
    print(f"    Mamba      : {mse_mb:.6f}")
    print(f"    FrameStack : {mse_fs:.6f}")

    # ── Metric 3 — Probe R² ────────────────────────────────────────────
    print("\n[4] Probe R² (Transformer)")
    r2_tr = probe_r2(model_tr, f, frame_ep, frame_loc, ep_offset,
                     embed_tr, seed=PROBE_SEED, label="Transformer")
    print("\n[5] Probe R² (Mamba)")
    r2_mb = probe_r2(model_mb, f, frame_ep, frame_loc, ep_offset,
                     embed_mb, seed=PROBE_SEED, label="Mamba")
    print("\n[6] Probe R² (FrameStack)")
    r2_fs = probe_r2(model_fs, f, frame_ep, frame_loc, ep_offset,
                     embed_fs, seed=PROBE_SEED, label="FrameStack")

    f.close()

    # ── Derived summary metrics ────────────────────────────────────────
    vel_r2_tr = float(np.mean(r2_tr[VEL_DIMS]))
    vel_r2_mb = float(np.mean(r2_mb[VEL_DIMS]))
    vel_r2_fs = float(np.mean(r2_fs[VEL_DIMS]))
    pos_r2_tr = float(np.mean(r2_tr[POS_DIMS]))
    pos_r2_mb = float(np.mean(r2_mb[POS_DIMS]))
    pos_r2_fs = float(np.mean(r2_fs[POS_DIMS]))

    # ── Print three-way comparison table ──────────────────────────────
    W = 78
    print("\n" + "─" * W)
    print(f"  {'Metric':<26} {'Transformer':>12} {'Mamba':>12} {'FrameStack':>12} {'FS vs TR':>10}")
    print("─" * W)

    def _chg(a, b, higher_better=True):
        d = b - a
        sign = ("▲" if (d > 0) == higher_better else "▼") if abs(d) > 1e-6 else "─"
        return f"{sign}{abs(d):.4f}"

    rows = [
        ("val/pred_loss",       mse_tr,     mse_mb,     mse_fs,     False),
        ("velocity R² (mean)",  vel_r2_tr,  vel_r2_mb,  vel_r2_fs,  True),
        ("position R² (tgt)",   pos_r2_tr,  pos_r2_mb,  pos_r2_fs,  True),
        ("pairwise L2 mean",    pw_mean_tr, pw_mean_mb, pw_mean_fs, None),
        ("pairwise L2 std",     pw_std_tr,  pw_std_mb,  pw_std_fs,  None),
    ]
    for label, v_tr, v_mb, v_fs, hb in rows:
        chg = _chg(v_tr, v_fs, hb) if hb is not None else f"{v_fs - v_tr:+.4f}"
        print(f"  {label:<26} {v_tr:>12.4f} {v_mb:>12.4f} {v_fs:>12.4f} {chg:>10}")
    print("─" * W)

    print(f"\n  Per-dimension R² breakdown:")
    print(f"  {'Dimension':<10} {'Transformer':>12} {'Mamba':>12} {'FrameStack':>12} "
          f"{'FS vs TR':>10}")
    print("  " + "─" * 58)
    for name, r_tr, r_mb, r_fs in zip(STATE_NAMES, r2_tr, r2_mb, r2_fs):
        tag = "[VEL]" if name in ("cue_vx", "cue_vy", "tgt_vx", "tgt_vy") else "[POS]"
        print(f"  {name:<10} {r_tr:>12.4f} {r_mb:>12.4f} {r_fs:>12.4f} "
              f"{r_fs - r_tr:>+10.4f}  {tag}")

    # ── Save PNG figure ────────────────────────────────────────────────
    _save_figure(
        r2_tr, r2_mb, r2_fs,
        mse_tr, mse_mb, mse_fs,
        vel_r2_tr, vel_r2_mb, vel_r2_fs,
        pos_r2_tr, pos_r2_mb, pos_r2_fs,
        pw_mean_tr, pw_mean_mb, pw_mean_fs,
        pw_std_tr,  pw_std_mb,  pw_std_fs,
    )
    print(f"\n  ✓ Saved comparison figure → {OUT_PNG}")


def _save_figure(
    r2_tr, r2_mb, r2_fs,
    mse_tr, mse_mb, mse_fs,
    vel_r2_tr, vel_r2_mb, vel_r2_fs,
    pos_r2_tr, pos_r2_mb, pos_r2_fs,
    pw_mean_tr, pw_mean_mb, pw_mean_fs,
    pw_std_tr,  pw_std_mb,  pw_std_fs,
):
    COLORS = {"Transformer": "#3498db", "Mamba": "#e74c3c", "FrameStack": "#2ecc71"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "LeWM Billiards — Three-Way Comparison: Transformer vs Mamba vs Frame Stacking",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Summary table (top) ──────────────────────────────────────────
    ax_tbl = fig.add_axes([0.02, 0.70, 0.96, 0.26])
    ax_tbl.axis("off")

    col_labels = ["Metric", "Transformer", "Mamba", "FrameStack",
                  "MB vs TR", "FS vs TR", "Winner"]
    table_data = []
    summary_rows = [
        ("val/pred_loss (↓)",   mse_tr,     mse_mb,     mse_fs,     False),
        ("velocity R² (↑)",     vel_r2_tr,  vel_r2_mb,  vel_r2_fs,  True),
        ("position R² (↑)",     pos_r2_tr,  pos_r2_mb,  pos_r2_fs,  True),
        ("pairwise L2 mean",    pw_mean_tr, pw_mean_mb, pw_mean_fs, None),
        ("pairwise L2 std",     pw_std_tr,  pw_std_mb,  pw_std_fs,  None),
    ]
    for label, v_tr, v_mb, v_fs, hb in summary_rows:
        if hb is None:
            d_mb, d_fs = v_mb - v_tr, v_fs - v_tr
            winner = "—"
        elif hb:
            d_mb = v_mb - v_tr
            d_fs = v_fs - v_tr
            best_v = max(v_tr, v_mb, v_fs)
            winner = (["Transformer", "Mamba", "FrameStack"]
                      [[v_tr, v_mb, v_fs].index(best_v)])
        else:
            d_mb = v_mb - v_tr
            d_fs = v_fs - v_tr
            best_v = min(v_tr, v_mb, v_fs)
            winner = (["Transformer", "Mamba", "FrameStack"]
                      [[v_tr, v_mb, v_fs].index(best_v)])
        table_data.append([
            label, f"{v_tr:.4f}", f"{v_mb:.4f}", f"{v_fs:.4f}",
            f"{d_mb:+.4f}", f"{d_fs:+.4f}", winner,
        ])

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # highlight velocity R² row
    for j in range(len(col_labels)):
        tbl[2, j].set_facecolor("#fef9c3")

    # ── Per-dimension R² bar chart (bottom-left) ──────────────────────
    ax_r2 = fig.add_axes([0.04, 0.06, 0.55, 0.58])
    x = np.arange(STATE_DIM)
    w = 0.25
    ax_r2.bar(x - w,   r2_tr, w, label="Transformer", color=COLORS["Transformer"], alpha=0.85)
    ax_r2.bar(x,       r2_mb, w, label="Mamba",        color=COLORS["Mamba"],       alpha=0.85)
    ax_r2.bar(x + w,   r2_fs, w, label="FrameStack",   color=COLORS["FrameStack"],  alpha=0.85)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(STATE_NAMES, rotation=35, ha="right", fontsize=9)
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Per-dimension Probe R²", fontsize=11)
    ax_r2.axhline(0, color="black", linewidth=0.7)
    ax_r2.set_ylim(-0.15, 1.05)
    for xi in VEL_DIMS:
        ax_r2.axvspan(xi - 0.5, xi + 0.5, color="gold", alpha=0.15, zorder=0)
    gold_patch = mpatches.Patch(color="gold", alpha=0.4, label="Velocity dims")
    handles, labels_ = ax_r2.get_legend_handles_labels()
    ax_r2.legend(handles + [gold_patch], labels_ + ["Velocity dims"], fontsize=8)

    # ── Velocity + Position R² highlight (bottom-right) ──────────────
    ax_vel = fig.add_axes([0.66, 0.06, 0.31, 0.58])
    categories = ["Velocity R²\n(mean cue+tgt)", "Position R²\n(tgt_x/y mean)"]
    vals = {
        "Transformer": [vel_r2_tr, pos_r2_tr],
        "Mamba":       [vel_r2_mb, pos_r2_mb],
        "FrameStack":  [vel_r2_fs, pos_r2_fs],
    }
    xv = np.arange(len(categories))
    offsets = [-0.27, 0, 0.27]
    for (name, vs), off_ in zip(vals.items(), offsets):
        bars = ax_vel.bar(xv + off_, vs, 0.25, label=name,
                          color=COLORS[name], alpha=0.85)
        for rect, val in zip(bars, vs):
            ax_vel.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )
    ax_vel.set_xticks(xv)
    ax_vel.set_xticklabels(categories, fontsize=9)
    ax_vel.set_ylabel("R²")
    ax_vel.set_title("Velocity vs Position R²\n(hypothesis check)", fontsize=11)
    ax_vel.set_ylim(0, 1.12)
    ax_vel.legend(fontsize=8)

    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

