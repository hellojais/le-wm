"""export_paper_figures.py

Generates publication-quality figures for the LeWM billiards paper.
Saves each figure as both PNG (300 DPI) and PDF (vector) in:
    experiments/billiards/figures/

Figures produced:
    fig1_training_curves.png / .pdf
    fig2_tsne.png / .pdf
    fig3_planning_combo_a.png / .pdf
    fig4_planning_combo_b.png / .pdf

Usage:
    uv run python experiments/billiards/export_paper_figures.py
"""

import os
import sys
import numpy as np
import torch
import h5py
import hdf5plugin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

# ── headless SDL for pygame ─────────────────────────────────────────────────
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # le-wm root (for jepa.py)
from game import BilliardsEnv, POCKET_POSITIONS, POCKET_RADIUS

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── shared paths ────────────────────────────────────────────────────────────
CKPT_PATH    = Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"
DEVICE       = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── shared save helper ──────────────────────────────────────────────────────
def save_figure(fig, stem: str):
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{stem}.{ext}"
        dpi = 300 if ext == "png" else None   # PDF is vector — dpi irrelevant
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curves():
    print("\n[1/4] Training curves …")

    orig_val_pred_loss = [
        0.101, 0.063, 0.074, 0.034,
        0.019, 0.013, 0.011, 0.010,
        0.009, 0.009, 0.009, 0.010,
        0.010, 0.011, 0.010, 0.009,
        0.009, 0.008, 0.007, 0.007,
    ]
    small_val_pred_loss = [
        0.0344, 0.0120, 0.0059, 0.0040,
        0.0042, 0.0035, 0.0104, 0.0028,
        0.0033, 0.0036,
    ]

    orig_epochs  = list(range(1, len(orig_val_pred_loss)  + 1))
    small_epochs = list(range(1, len(small_val_pred_loss) + 1))
    orig_best_epoch  = int(np.argmin(orig_val_pred_loss))  + 1
    small_best_epoch = int(np.argmin(small_val_pred_loss)) + 1
    orig_best_val    = min(orig_val_pred_loss)
    small_best_val   = min(small_val_pred_loss)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(orig_epochs, orig_val_pred_loss,
            color="steelblue", linewidth=2, marker="o", markersize=4,
            label="Original (embed\_dim=192, λ=0.09, 20 epochs)")
    ax.plot(small_epochs, small_val_pred_loss,
            color="crimson", linewidth=2, marker="o", markersize=4,
            label="Small (embed\_dim=32, λ=0.01, 10 epochs)")

    ax.plot(orig_best_epoch, orig_best_val,
            marker="*", color="steelblue", markersize=14, zorder=5,
            label=f"Original best (epoch {orig_best_epoch}, loss={orig_best_val:.4f})")
    ax.plot(small_best_epoch, small_best_val,
            marker="*", color="crimson", markersize=14, zorder=5,
            label=f"Small best (epoch {small_best_epoch}, loss={small_best_val:.4f})")

    ax.axvline(orig_best_epoch,  color="steelblue", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(small_best_epoch, color="crimson",   linestyle="--", linewidth=1, alpha=0.5)

    ratio = orig_best_val / small_best_val
    ax.annotate(
        f"{ratio:.1f}× better",
        xy=(small_best_epoch, small_best_val),
        xytext=(small_best_epoch + 1.5, small_best_val * 3.0),
        fontsize=12, fontweight="bold", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Prediction Loss (log scale)", fontsize=12)
    ax.set_xticks(range(1, max(len(orig_val_pred_loss), len(small_val_pred_loss)) + 1))
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    save_figure(fig, "fig1_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — t-SNE
# ─────────────────────────────────────────────────────────────────────────────
def fig_tsne():
    print("\n[2/4] t-SNE …")

    # load model
    model = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.eval().to(DEVICE)

    # sample 2000 frames
    with h5py.File(DATASET_PATH, "r") as f:
        total = f["pixels"].shape[0]
        idx   = np.sort(np.linspace(0, total - 1, 2000, dtype=int))
        frames = f["pixels"][idx]   # (N,H,W,3) uint8
        states = f["state"][idx].astype(np.float32)

    # extract embeddings
    frames_f = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    embs = []
    with torch.no_grad():
        for i in range(0, len(frames_f), 64):
            b = frames_f[i:i+64].to(DEVICE)
            out = model.encode({"pixels": b.unsqueeze(1)})
            e = out["emb"]
            if e.ndim == 3:
                e = e[:, 0, :]
            embs.append(e.cpu().numpy())
    embeddings = np.concatenate(embs, axis=0)

    # state features
    target_x     = states[:, 4]
    target_y     = states[:, 5]
    pocket_x     = states[:, 8]
    pocket_y     = states[:, 9]
    dist_pocket  = np.sqrt((target_x - pocket_x)**2 + (target_y - pocket_y)**2)
    emb_mag      = np.linalg.norm(embeddings, axis=1)
    near_mask    = dist_pocket < 30.0

    # t-SNE
    print("  Running t-SNE …")
    tsne_coords = TSNE(n_components=2, perplexity=30, max_iter=1000,
                       random_state=42, verbose=0).fit_transform(embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        (axes[0, 0], target_x,   "RdYlGn",   "Target ball X (px)"),
        (axes[0, 1], target_y,   "RdYlGn",   "Target ball Y (px)"),
        (axes[1, 0], dist_pocket,"RdYlGn_r", "Distance to nearest pocket (px)"),
        (axes[1, 1], emb_mag,    "viridis",  "Embedding L2 magnitude"),
    ]

    for ax, vals, cmap, cbar_label in configs:
        sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                        c=vals, cmap=cmap, s=5, alpha=0.7, rasterized=True)
        if "Distance" in cbar_label and near_mask.sum() > 0:
            ax.scatter(tsne_coords[near_mask, 0], tsne_coords[near_mask, 1],
                       s=20, facecolors="none", edgecolors="black", linewidths=0.8,
                       label=f"Near pocket (<30 px, n={near_mask.sum()})", zorder=5)
            ax.legend(fontsize=10, loc="upper right")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(cbar_label, fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.set_xlabel("t-SNE dim 1", fontsize=11)
        ax.set_ylabel("t-SNE dim 2", fontsize=11)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_figure(fig, "fig2_tsne")


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING HELPERS (shared by fig 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────
PLAN_HORIZON  = 20
EXEC_PER_PLAN = 5
MAX_EVAL_STEPS = 300
NUM_SAMPLES   = 500
N_CEM_ITERS   = 20
TOPK          = 50
ACT_LO, ACT_HI = -30.0, 30.0


def _min_dist(pos):
    return float(np.linalg.norm(POCKET_POSITIONS - pos, axis=1).min())


def _make_env(state):
    env = BilliardsEnv(render_mode="rgb_array")
    env.reset()
    env.cue_pos        = state[0:2].astype(np.float32).copy()
    env.cue_vel        = state[2:4].astype(np.float32).copy()
    env.target_pos     = state[4:6].astype(np.float32).copy()
    env.target_vel     = state[6:8].astype(np.float32).copy()
    env._step_count    = 0
    env._target_potted = False
    return env


def _simulate(start_state, actions):
    env = _make_env(start_state)
    best = _min_dist(env.target_pos)
    for act in actions:
        _, _, t, tr, _ = env.step(act)
        d = _min_dist(env.target_pos)
        if d < best:
            best = d
        if t or tr:
            break
    return best


def _plan_cem(start_state, warm_mean=None):
    act_dim = 2
    if warm_mean is not None:
        mean = warm_mean.copy()
        std  = np.full((PLAN_HORIZON, act_dim), 5.0, dtype=np.float32)
    else:
        mean = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)
        std  = np.full((PLAN_HORIZON, act_dim), 15.0, dtype=np.float32)

    best_cost = float("inf")
    best_acts = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)

    for _ in range(N_CEM_ITERS):
        noise   = np.random.randn(NUM_SAMPLES, PLAN_HORIZON, act_dim).astype(np.float32)
        samples = (mean[None] + std[None] * noise).clip(ACT_LO, ACT_HI)
        costs   = np.array([_simulate(start_state, samples[i]) for i in range(NUM_SAMPLES)])
        elite   = np.argpartition(costs, TOPK)[:TOPK]
        mean    = samples[elite].mean(0)
        std     = samples[elite].std(0).clip(min=0.5)
        it_best = float(costs[elite].min())
        if it_best < best_cost:
            best_cost = it_best
            best_acts = samples[elite[costs[elite].argmin()]].copy()

    return best_acts, best_cost, mean


def run_planning(start_state, seed=1):
    np.random.seed(seed)
    env        = _make_env(start_state)
    start_frame = env.render()
    all_frames  = [start_frame]
    success     = False
    best_dist   = float("inf")
    total_steps = 0
    warm_mean   = None
    n_plans     = (MAX_EVAL_STEPS + EXEC_PER_PLAN - 1) // EXEC_PER_PLAN

    for _ in range(n_plans):
        if MAX_EVAL_STEPS - total_steps <= 0:
            break
        cur_state = np.concatenate([
            env.cue_pos, env.cue_vel, env.target_pos, env.target_vel, np.zeros(2)])
        acts, plan_best, final_mean = _plan_cem(cur_state, warm_mean)
        if plan_best < best_dist:
            best_dist = plan_best
        shift     = final_mean[EXEC_PER_PLAN:]
        warm_mean = np.concatenate([shift, np.zeros((EXEC_PER_PLAN, 2), dtype=np.float32)])
        done = False
        for act in acts[:min(EXEC_PER_PLAN, MAX_EVAL_STEPS - total_steps)]:
            _, _, term, trunc, _ = env.step(act)
            all_frames.append(env.render())
            d = _min_dist(env.target_pos)
            if d < best_dist:
                best_dist = d
            total_steps += 1
            if term:
                success = True
                done    = True
                break
            if trunc:
                done = True
                break
        if done:
            break

    env.close()
    print(f"    {'SUCCESS ✓' if success else 'FAILURE ✗'}  "
          f"steps={total_steps}  best_dist={best_dist:.2f}")
    return {
        "start_frame": start_frame,
        "final_frame": all_frames[-1],
        "success":     success,
        "total_steps": total_steps,
        "best_dist":   best_dist,
    }


def _planning_panel(fig, axes, result, goal_frame, combo_label):
    """Fill one row (3 axes) with start | goal | result."""
    success = result["success"]
    status  = "SUCCESS" if success else "FAILURE"
    c_ok    = "#2d6a4f" if success else "#9b2226"

    panels = [
        (result["start_frame"], "Start",              "white"),
        (goal_frame,            "Goal (episode end)", "white"),
        (result["final_frame"], f"{status}  ({result['total_steps']} steps)", c_ok),
    ]
    for ax, (img, title, tc) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, color=tc, pad=4)
        ax.axis("off")
    axes[0].set_ylabel(combo_label, fontsize=12, labelpad=8)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Planning Combo A
# ─────────────────────────────────────────────────────────────────────────────
def fig_planning_combo_a(f, ep_len, ep_offset):
    print("\n[3/4] Planning Combo A (ep 500) …")
    s           = int(ep_offset[500])
    start_state = f["state"][s]
    goal_frame  = f["pixels"][s + int(ep_len[500]) - 1]

    result = run_planning(start_state, seed=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"wspace": 0.04})
    _planning_panel(fig, axes, result, goal_frame,
                    "Combo A\n(same episode)")
    plt.tight_layout()
    save_figure(fig, "fig3_planning_combo_a")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Planning Combo B
# ─────────────────────────────────────────────────────────────────────────────
def fig_planning_combo_b(f, ep_len, ep_offset):
    print("\n[4/4] Planning Combo B (ep 100 start / ep 3000 goal) …")
    s_start     = int(ep_offset[100])
    start_state = f["state"][s_start]
    s_goal      = int(ep_offset[3000])
    goal_frame  = f["pixels"][s_goal + int(ep_len[3000]) - 1]

    result = run_planning(start_state, seed=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             gridspec_kw={"wspace": 0.04})
    _planning_panel(fig, axes, result, goal_frame,
                    "Combo B\n(novel cross-episode)")
    plt.tight_layout()
    save_figure(fig, "fig4_planning_combo_b")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  export_paper_figures.py")
    print(f"  Output directory: {OUT_DIR}")
    print("=" * 60)

    fig_training_curves()
    fig_tsne()

    print("\n  Loading dataset for planning figures …")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]

    fig_planning_combo_a(f, ep_len, ep_offset)
    fig_planning_combo_b(f, ep_len, ep_offset)

    f.close()

    print("\n" + "=" * 60)
    print("  All figures saved to:")
    for p in sorted(OUT_DIR.iterdir()):
        kb = p.stat().st_size / 1024
        print(f"    {p.name:45s}  {kb:6.0f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
