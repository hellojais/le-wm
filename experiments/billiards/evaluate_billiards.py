"""evaluate_billiards.py

Evaluate a trained LeWM world model on the billiards task.

Run from the le-wm directory:
    python evaluate_billiards.py

Two evaluations are run:
  Combo A — Same episode:      ep 500 frame 0 (start) → ep 500 last frame (goal)
  Combo B — Novel cross-ep:    ep 100 frame 0 (start) → ep 3000 last frame (goal)
"""

import os
import sys
from pathlib import Path

# ── Headless SDL so pygame never tries to open a display window ───────────────
os.environ["SDL_VIDEODRIVER"]  = "dummy"
os.environ["SDL_AUDIODRIVER"]  = "dummy"

import numpy as np
import torch
import torch.nn.functional as F
import hdf5plugin  # must be imported before h5py to register codecs
import h5py
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI required
import matplotlib.pyplot as plt

# Add billiards-worldmodel to path so we can import game.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
from game import BilliardsEnv

# ── Configuration ─────────────────────────────────────────────────────────────
# lewm_small_epoch_8_object.ckpt = epoch 7 (0-indexed), best val/pred_loss=0.00284
# embed_dim=32, sigreg_weight=0.01 — 8.5× better prediction than original model
CHECKPOINT   = Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"

DEVICE       = torch.device("mps") if torch.backends.mps.is_available() \
               else torch.device("cpu")

HISTORY_SIZE   = 3    # wm.history_size from billiards_small.yaml
PLAN_HORIZON   = 10   # steps to plan ahead each CEM call
EXEC_PER_PLAN  = 3    # real env steps to execute before re-planning (MPC)
MAX_EVAL_STEPS = 150  # total execution budget per evaluation
NUM_SAMPLES    = 1000 # CEM population size
N_CEM_ITERS    = 20   # CEM refinement iterations
TOPK           = 50   # elite samples kept per CEM iteration

# CEM action clamp in normalised space.
# Dataset std≈1, so ±3 covers ~99.7% of the action distribution.
# The new model was trained on these normalised actions — keep clamp at 3.
CEM_CLAMP      = 3.0

# ImageNet normalisation constants (match train.py preprocessing)
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 numpy → CHW float32 ImageNet-normalised tensor."""
    t = torch.from_numpy(frame_hwc.copy()).float() / 255.0  # (H, W, C)
    t = t.permute(2, 0, 1)                                  # (C, H, W)
    return (t - _MEAN) / _STD


def frames_to_tensor(frames: list) -> torch.Tensor:
    """List of HWC ndarrays → (T, C, H, W) float32 tensor."""
    return torch.stack([preprocess_frame(f) for f in frames])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"\n[1] Loading checkpoint: {CHECKPOINT.name}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Loaded  ({n_params:,} params, device={DEVICE})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    print(f"\n[2] Loading dataset: {DATASET_PATH.name}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {len(ep_len)} episodes, {f['pixels'].shape[0]:,} frames")
    return f, ep_len, ep_offset


def get_episode(f, ep_len, ep_offset, ep_idx):
    """Return pixels (T,H,W,C), state (T,10), action (T,2) for one episode."""
    s = ep_offset[ep_idx]
    e = s + ep_len[ep_idx]
    return {
        "pixels": f["pixels"][s:e],
        "state":  f["state"][s:e],
        "action": f["action"][s:e],
    }


def action_stats(f):
    """Compute action mean/std from the full dataset (matches train.py normalisation)."""
    acts = f["action"][:]
    mean = acts.mean(0).astype(np.float32)
    std  = acts.std(0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)

    print(f"     action mean  = {mean}")
    print(f"     action std   = {std}")
    print(f"     action range = [{acts.min(axis=0)}, {acts.max(axis=0)}]")
    print(f"     CEM clamp    = ±{CEM_CLAMP}  (normalised space)")
    cem_env_lo = -CEM_CLAMP * std + mean
    cem_env_hi =  CEM_CLAMP * std + mean
    print(f"     CEM env range dim0: [{cem_env_lo[0]:.2f}, {cem_env_hi[0]:.2f}]")
    print(f"     CEM env range dim1: [{cem_env_lo[1]:.2f}, {cem_env_hi[1]:.2f}]")
    coverage = (cem_env_hi - cem_env_lo) / (acts.max(axis=0) - acts.min(axis=0)) * 100
    print(f"     Coverage of dataset range: {coverage[0]:.0f}% / {coverage[1]:.0f}%")
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CEM PLANNING
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def plan_cem(model, ctx_frames: list, goal_frame: np.ndarray,
             act_mean_np: np.ndarray, act_std_np: np.ndarray,
             hist_act_norm: np.ndarray | None = None,
             warm_mean_norm: np.ndarray | None = None):
    """
    Cross-Entropy Method planning.

    Args:
        ctx_frames:     HISTORY_SIZE HWC numpy frames (start context)
        goal_frame:     1 HWC numpy frame (desired goal)
        act_mean_np:    (2,) action mean for denormalisation
        act_std_np:     (2,) action std  for denormalisation
        hist_act_norm:  (HISTORY_SIZE, 2) normalised past actions (or None → zeros)
        warm_mean_norm: (PLAN_HORIZON, 2) normalised warm-start mean (or None)

    Returns:
        best_actions_env:  (PLAN_HORIZON, 2) actions in environment units
        best_cost:         scalar goal-embedding MSE of the best plan
        cem_mean_norm_np:  (PLAN_HORIZON, 2) final CEM mean in normalised space
                           (pass as warm_mean_norm for the next re-plan)
    """
    B, S, H, act_dim = 1, NUM_SAMPLES, HISTORY_SIZE, 2

    # ── Preprocess frames ────────────────────────────────────────────────────
    ctx  = frames_to_tensor(ctx_frames).to(DEVICE)          # (H, C, h, w)
    goal = preprocess_frame(goal_frame).to(DEVICE)          # (C, h, w)

    # ── Pre-encode goal once (avoids re-encoding every CEM step) ────────────
    goal_for_enc = goal.unsqueeze(0).unsqueeze(0)           # (1, 1, C, h, w)
    goal_emb_raw = model.encode({"pixels": goal_for_enc})["emb"]  # (1, 1, D)
    # Expand to (B, S, 1, D) for broadcasting in cost computation
    goal_emb_bs  = goal_emb_raw.unsqueeze(1).expand(B, S, 1, -1)  # (1, S, 1, D)

    # ── Context frames: (B, S, H, C, h, w) ──────────────────────────────────
    ctx_bst = ctx.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1, -1)

    # ── History actions: real executed actions or zeros ───────────────────────
    if hist_act_norm is not None:
        ha = torch.from_numpy(hist_act_norm.astype(np.float32)).to(DEVICE)  # (H, 2)
        hist_act = ha.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)        # (B,S,H,2)
    else:
        hist_act = torch.zeros(B, S, H, act_dim, device=DEVICE)

    # ── CEM distribution: warm-start from previous plan if available ──────────
    if warm_mean_norm is not None:
        cem_mean = torch.from_numpy(warm_mean_norm.astype(np.float32)).to(DEVICE)
        cem_std  = torch.ones(PLAN_HORIZON, act_dim, device=DEVICE) * 0.5  # tighter
    else:
        cem_mean = torch.zeros(PLAN_HORIZON, act_dim, device=DEVICE)
        cem_std  = torch.ones( PLAN_HORIZON, act_dim, device=DEVICE)

    best_cost        = float("inf")
    best_actions_env = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)

    print(f"    CEM: {N_CEM_ITERS} iters × {S} samples × horizon={PLAN_HORIZON}  clamp=±{CEM_CLAMP}")

    for it in range(N_CEM_ITERS):
        # Sample (S, PLAN_HORIZON, 2) future actions in normalised space
        noise   = torch.randn(S, PLAN_HORIZON, act_dim, device=DEVICE)
        fut_act = (cem_mean + cem_std * noise).clamp(-CEM_CLAMP, CEM_CLAMP)  # (S, PLAN_HORIZON, 2)

        # Full action sequence (B, S, H+HORIZON, 2)
        action_seq = torch.cat([hist_act, fut_act.unsqueeze(0)], dim=2)

        # Rollout: in-place populates info["predicted_emb"] (B, S, H+HORIZON+1, D)
        info = {"pixels": ctx_bst}
        model.rollout(info, action_seq, history_size=HISTORY_SIZE)

        pred_emb = info["predicted_emb"]               # (B, S, H+HORIZON+1, D)

        # Cost = MSE between last predicted step and goal embedding
        # pred: (B, S, 1, D)   goal: (B, S, 1, D)
        cost = F.mse_loss(
            pred_emb[:, :, -1:, :],
            goal_emb_bs,
            reduction="none",
        ).sum(-1).squeeze(-1)                           # (B, S)

        costs_s = cost[0]                               # (S,)

        # Update distribution with top-k elites
        _, top_idx = torch.topk(-costs_s, k=TOPK)
        elites   = fut_act[top_idx]                     # (TOPK, HORIZON, 2)
        cem_mean = elites.mean(0)                       # (HORIZON, 2)
        cem_std  = elites.std(0).clamp(min=1e-3)

        it_best = costs_s.min().item()
        mean_act_mag = fut_act[top_idx].abs().mean().item()
        print(f"      iter {it+1:2d}/{N_CEM_ITERS}  best_cost = {it_best:.4f}  "
              f"elite_act_mag = {mean_act_mag:.3f}")

        if it_best < best_cost:
            best_cost = it_best
            # Denormalise best actions → environment units
            best_norm_np     = cem_mean.cpu().numpy()               # (PLAN_HORIZON, 2)
            best_actions_env = best_norm_np * act_std_np + act_mean_np

    print(f"    ✓ Planning done  best_cost = {best_cost:.4f}  "
          f"best_act_mag (env) = {np.abs(best_actions_env).mean():.3f}")
    return best_actions_env, best_cost, cem_mean.cpu().numpy()


# (execute() removed — MPC loop is now inside evaluate())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALISE AND SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(label, start_frame, goal_frame, exec_frames, success, cost, out_path):
    """Save a 3-panel figure: start | goal | execution strip."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    status = "SUCCESS ✓" if success else "FAILURE ✗"
    color  = "green" if success else "crimson"
    fig.suptitle(
        f"{label}\n{status}   |   Goal-embedding distance: {cost:.6f}",
        fontsize=12, fontweight="bold", color=color,
    )

    axes[0].imshow(start_frame)
    axes[0].set_title("Start frame")
    axes[0].axis("off")

    axes[1].imshow(goal_frame)
    axes[1].set_title("Goal frame")
    axes[1].axis("off")

    # Right panel: first / middle / last execution frames side-by-side
    n   = len(exec_frames)
    mid = exec_frames[n // 2]
    strip = np.concatenate([exec_frames[0], mid, exec_frames[-1]], axis=1)
    axes[2].imshow(strip)
    axes[2].set_title(f"Execution: start | mid | end  ({n} steps total)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE FOR ONE (start, goal) COMBINATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, f, ep_len, ep_offset, act_mean, act_std,
             label, start_ep, start_frame_idx, goal_ep, out_path):
    """
    Receding-horizon MPC evaluation.

    Each iteration: plan PLAN_HORIZON steps with CEM (warm-started from previous
    plan), execute EXEC_PER_PLAN real env steps, re-encode from real frames,
    re-plan.  Runs until target is potted (SUCCESS) or MAX_EVAL_STEPS spent.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Start: episode {start_ep} frame {start_frame_idx}")
    print(f"  Goal:  episode {goal_ep} last frame")
    print(f"  MPC: plan={PLAN_HORIZON} steps, exec={EXEC_PER_PLAN}/plan, budget={MAX_EVAL_STEPS}")
    print(f"{'─' * 60}")

    start_data = get_episode(f, ep_len, ep_offset, start_ep)
    goal_data  = get_episode(f, ep_len, ep_offset, goal_ep)

    # Build HISTORY_SIZE-frame context ending at start_frame_idx
    ctx_indices = [max(0, start_frame_idx - HISTORY_SIZE + 1 + i)
                   for i in range(HISTORY_SIZE)]
    ctx_frames  = [start_data["pixels"][i] for i in ctx_indices]
    start_state = start_data["state"][start_frame_idx]

    # Use a frame 4 steps before episode end as goal — avoids the pocket-sink
    # animation that makes the last frame visually unusual.
    goal_idx   = max(0, len(goal_data["pixels"]) - 4)
    goal_frame = goal_data["pixels"][goal_idx]

    # Initialise environment to the recorded start state
    env = BilliardsEnv(render_mode="rgb_array")
    env.reset()  # initialises pygame surfaces
    env.cue_pos    = start_state[0:2].astype(np.float32).copy()
    env.cue_vel    = start_state[2:4].astype(np.float32).copy()
    env.target_pos = start_state[4:6].astype(np.float32).copy()
    env.target_vel = start_state[6:8].astype(np.float32).copy()
    env._step_count    = 0
    env._target_potted = False

    all_frames       = [env.render()]  # real rendered frames (HWC uint8)
    exec_norm_buffer = []              # rolling buffer of normalised executed actions
    cem_warm_mean    = None            # warm-start for next CEM call
    success          = False
    best_cost        = float("inf")
    total_steps      = 0
    first_plan       = True
    n_plans          = (MAX_EVAL_STEPS + EXEC_PER_PLAN - 1) // EXEC_PER_PLAN
    act_dim          = 2

    for plan_idx in range(n_plans):
        remaining = MAX_EVAL_STEPS - total_steps
        if remaining <= 0:
            break

        # Build history action context: last HISTORY_SIZE normalised actions
        if len(exec_norm_buffer) > 0:
            h = np.array(exec_norm_buffer[-HISTORY_SIZE:])  # up to (H, 2)
            if len(h) < HISTORY_SIZE:
                h = np.pad(h, ((HISTORY_SIZE - len(h), 0), (0, 0)))
            hist_arr = h.astype(np.float32)  # (HISTORY_SIZE, 2)
        else:
            hist_arr = None

        # Embedding distance to goal at current real state
        with torch.inference_mode():
            cur_img  = torch.stack([preprocess_frame(ctx_frames[-1])]).unsqueeze(0).to(DEVICE)
            cur_emb  = model.encode({"pixels": cur_img})["emb"][0, 0]
            goal_img = torch.stack([preprocess_frame(goal_frame)]).unsqueeze(0).to(DEVICE)
            goal_emb = model.encode({"pixels": goal_img})["emb"][0, 0]
            emb_dist = (cur_emb - goal_emb).norm().item()

        print(f"\n[3] CEM plan {plan_idx + 1}/{n_plans}  "
              f"(step {total_steps}/{MAX_EVAL_STEPS})  "
              f"emb_dist_to_goal={emb_dist:.3f} ...")
        plan_actions, cost, plan_norm = plan_cem(
            model, ctx_frames, goal_frame, act_mean, act_std,
            hist_act_norm=hist_arr,
            warm_mean_norm=cem_warm_mean,
        )
        best_cost = min(best_cost, cost)

        if first_plan:
            print("    First plan actions (env units):")
            for i, a in enumerate(plan_actions[:EXEC_PER_PLAN]):
                print(f"      step {i}: [{a[0]:+.3f}, {a[1]:+.3f}]")
            first_plan = False

        # Warm-start next plan: shift plan_norm by EXEC_PER_PLAN, pad end with 0s
        shift = plan_norm[EXEC_PER_PLAN:]  # (PLAN_HORIZON - EXEC_PER_PLAN, 2)
        cem_warm_mean = np.concatenate(
            [shift, np.zeros((PLAN_HORIZON - len(shift), act_dim), dtype=np.float32)], axis=0
        )

        # Execute the first EXEC_PER_PLAN actions from this plan
        exec_slice = plan_actions[:min(EXEC_PER_PLAN, remaining)]
        done = False
        for act in exec_slice:
            _, _, terminated, truncated, _ = env.step(act)
            all_frames.append(env.render())
            # Store normalised executed action for history context
            exec_norm_buffer.append((act - act_mean) / np.maximum(act_std, 1e-6))
            total_steps += 1
            if terminated:
                success = True
                done    = True
                break
            if truncated:
                done = True
                break

        # Update context: last HISTORY_SIZE real rendered frames
        ctx_frames = [all_frames[max(0, len(all_frames) - HISTORY_SIZE + i)]
                      for i in range(HISTORY_SIZE)]

        if done:
            break

    env.close()

    print(f"\n[4] Executed {total_steps} total steps  →  "
          f"{'SUCCESS ✓' if success else 'FAILURE ✗'}")

    print(f"\n[5] Saving figure ...")
    save_figure(
        label,
        start_data["pixels"][start_frame_idx],
        goal_frame,
        all_frames,
        success, best_cost,
        out_path,
    )
    return success, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LeWM Billiards Evaluation")
    print(f"  Device : {DEVICE}")
    print(f"  Checkpoint: {CHECKPOINT.name}")
    print("=" * 60)

    model                = load_model()
    f, ep_len, ep_offset = load_dataset()

    print("\n[2b] Computing action normalisation stats ...")
    act_mean, act_std = action_stats(f)
    print(f"     action mean = {act_mean}   std = {act_std}")

    results = []

    # ── Combo A: same episode ────────────────────────────────────────────────
    s, c = evaluate(
        model, f, ep_len, ep_offset, act_mean, act_std,
        label           = "Combo A — Same episode (ep 500 start → ep 500 goal)",
        start_ep        = 500,
        start_frame_idx = 0,
        goal_ep         = 500,
        out_path        = Path(__file__).parent / "evaluation_result_A.png",
    )
    results.append(("Combo A (same ep)", s, c))

    # ── Combo B: novel cross-episode ─────────────────────────────────────────
    s, c = evaluate(
        model, f, ep_len, ep_offset, act_mean, act_std,
        label           = "Combo B — Novel cross-episode (ep 100 start → ep 3000 goal)",
        start_ep        = 100,
        start_frame_idx = 0,
        goal_ep         = 3000,
        out_path        = Path(__file__).parent / "evaluation_result_B.png",
    )
    results.append(("Combo B (cross-ep)", s, c))

    f.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, success, cost in results:
        status = "SUCCESS ✓" if success else "FAILURE ✗"
        print(f"  {name:30s}  {status}   goal-emb dist = {cost:.6f}")
    print("=" * 60)
    print("\n[6] Done.")
    print("    Results saved as:")
    print("      evaluation_result_A.png  (same episode)")
    print("      evaluation_result_B.png  (novel cross-episode)")


if __name__ == "__main__":
    main()
