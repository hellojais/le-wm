"""evaluate_billiards_state.py

State-based CEM planning for billiards.

Instead of using world-model embedding distance as cost, this script uses
the actual game simulator to evaluate action sequences:

  cost(sequence) = min distance from target ball to any pocket
                   across all simulated steps

This is model-free planning: CEM runs rollouts directly in the simulator
and picks the action sequence that moves the target ball closest to a pocket.

Two evaluations:
  Combo A — ep 500 frame 0 start, ep 500 target ball must reach pocket
  Combo B — ep 100 frame 0 start, ep 100 target ball must reach pocket
  (no "goal frame" needed — goal is always "pot the target ball")

Run from the le-wm directory:
    uv run python evaluate_billiards_state.py
"""

import copy
import os
import sys
from pathlib import Path

# Headless SDL
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import hdf5plugin  # must be before h5py
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
from game import BilliardsEnv, POCKET_POSITIONS, POCKET_RADIUS

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"

# CEM parameters
PLAN_HORIZON  = 20    # steps to simulate per candidate sequence
EXEC_PER_PLAN = 5     # real env steps to execute before re-planning
MAX_EVAL_STEPS = 300  # max total execution steps
NUM_SAMPLES   = 500   # CEM population size (parallelised via numpy)
N_CEM_ITERS   = 20    # CEM refinement iterations
TOPK          = 50    # elite samples kept per iteration

# Action space of BilliardsEnv: declared ±10, but dataset shows ±30
# We sample in the range that expert actions occupy
ACT_LO = -30.0
ACT_HI =  30.0

# Success: target ball within pocket radius of any pocket
SUCCESS_DIST = POCKET_RADIUS  # same check as env._check_pocket


# ─────────────────────────────────────────────────────────────────────────────
# DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    print(f"\n[1] Loading dataset: {DATASET_PATH.name}")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {len(ep_len)} episodes, {f['pixels'].shape[0]:,} frames")
    return f, ep_len, ep_offset


def get_episode(f, ep_len, ep_offset, ep_idx):
    s = ep_offset[ep_idx]
    e = s + ep_len[ep_idx]
    return {
        "pixels": f["pixels"][s:e],
        "state":  f["state"][s:e],
        "action": f["action"][s:e],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENV CLONING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_env_from_state(state: np.ndarray) -> BilliardsEnv:
    """Create a new BilliardsEnv initialised to the given 10-dim state vector."""
    env = BilliardsEnv(render_mode="rgb_array")
    env.reset()
    env.cue_pos    = state[0:2].astype(np.float32).copy()
    env.cue_vel    = state[2:4].astype(np.float32).copy()
    env.target_pos = state[4:6].astype(np.float32).copy()
    env.target_vel = state[6:8].astype(np.float32).copy()
    env._step_count    = 0
    env._target_potted = False
    return env


def clone_env(env: BilliardsEnv) -> BilliardsEnv:
    """Shallow-copy an env by reading out its state vector."""
    state = np.concatenate([
        env.cue_pos, env.cue_vel,
        env.target_pos, env.target_vel,
        np.zeros(2),  # nearest_pocket not needed for physics
    ])
    return make_env_from_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def min_dist_to_pocket(target_pos: np.ndarray) -> float:
    """Minimum Euclidean distance from target_pos to any pocket."""
    return float(np.linalg.norm(POCKET_POSITIONS - target_pos, axis=1).min())


def simulate_sequence(start_state: np.ndarray, actions: np.ndarray) -> float:
    """
    Simulate `actions` (PLAN_HORIZON, 2) from start_state.
    Returns the minimum distance from target ball to any pocket achieved
    across the trajectory (lower = better).
    """
    env = make_env_from_state(start_state)
    best_dist = min_dist_to_pocket(env.target_pos)

    for act in actions:
        _, _, terminated, truncated, _ = env.step(act)
        d = min_dist_to_pocket(env.target_pos)
        if d < best_dist:
            best_dist = d
        if terminated or truncated:
            break

    env.close()
    return best_dist


# ─────────────────────────────────────────────────────────────────────────────
# STATE-BASED CEM PLANNER
# ─────────────────────────────────────────────────────────────────────────────

def plan_cem_state(start_state: np.ndarray,
                   warm_mean: np.ndarray | None = None) -> tuple:
    """
    CEM planner using simulator-based cost.

    Args:
        start_state:  (10,) current env state
        warm_mean:    (PLAN_HORIZON, 2) warm-start mean or None

    Returns:
        best_actions:  (PLAN_HORIZON, 2) best action sequence
        best_cost:     float, min dist to pocket achieved
        final_mean:    (PLAN_HORIZON, 2) final CEM mean (for warm-starting)
    """
    act_dim = 2

    if warm_mean is not None:
        cem_mean = warm_mean.copy()
        cem_std  = np.ones((PLAN_HORIZON, act_dim), dtype=np.float32) * 5.0
    else:
        cem_mean = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)
        cem_std  = np.ones((PLAN_HORIZON, act_dim), dtype=np.float32) * 15.0

    best_cost    = float("inf")
    best_actions = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)

    print(f"    CEM: {N_CEM_ITERS} iters × {NUM_SAMPLES} samples × "
          f"horizon={PLAN_HORIZON}  action=[{ACT_LO}, {ACT_HI}]")

    for it in range(N_CEM_ITERS):
        # Sample (NUM_SAMPLES, PLAN_HORIZON, 2)
        noise    = np.random.randn(NUM_SAMPLES, PLAN_HORIZON, act_dim).astype(np.float32)
        samples  = (cem_mean[None] + cem_std[None] * noise).clip(ACT_LO, ACT_HI)

        # Evaluate each sample in the simulator
        costs = np.array([
            simulate_sequence(start_state, samples[i])
            for i in range(NUM_SAMPLES)
        ], dtype=np.float32)

        # Select top-k elites (lowest cost)
        elite_idx  = np.argpartition(costs, TOPK)[:TOPK]
        elites     = samples[elite_idx]   # (TOPK, PLAN_HORIZON, 2)
        cem_mean   = elites.mean(0)
        cem_std    = elites.std(0).clip(min=0.5)

        it_best = costs[elite_idx].min()
        best_sample_idx = elite_idx[costs[elite_idx].argmin()]
        mean_act_mag = np.abs(elites).mean()

        print(f"      iter {it+1:2d}/{N_CEM_ITERS}  "
              f"best_dist = {it_best:.2f}  "
              f"elite_act_mag = {mean_act_mag:.2f}")

        if it_best < best_cost:
            best_cost    = float(it_best)
            best_actions = samples[best_sample_idx].copy()

    print(f"    ✓ Planning done  best_dist_to_pocket = {best_cost:.2f}  "
          f"(pocket radius = {POCKET_RADIUS})")
    return best_actions, best_cost, cem_mean


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(label, start_frame, exec_frames, success, best_dist, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    status = "SUCCESS ✓" if success else "FAILURE ✗"
    color  = "green" if success else "crimson"
    fig.suptitle(
        f"{label}\n{status}   |   Best dist to pocket: {best_dist:.2f}  "
        f"(pocket radius = {POCKET_RADIUS})",
        fontsize=12, fontweight="bold", color=color,
    )

    axes[0].imshow(start_frame)
    axes[0].set_title("Start frame")
    axes[0].axis("off")

    n   = len(exec_frames)
    mid = exec_frames[n // 2]
    strip = np.concatenate([exec_frames[0], mid, exec_frames[-1]], axis=1)
    axes[1].imshow(strip)
    axes[1].set_title(f"Execution: start | mid | end  ({n} steps)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE EPISODE START
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(f, ep_len, ep_offset, label, start_ep, start_frame_idx, out_path):
    """
    MPC evaluation using state-based CEM.
    Goal: pot the target ball into any pocket.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Start: episode {start_ep} frame {start_frame_idx}")
    print(f"  MPC: plan={PLAN_HORIZON}, exec={EXEC_PER_PLAN}/plan, budget={MAX_EVAL_STEPS}")
    print(f"{'─' * 60}")

    start_data  = get_episode(f, ep_len, ep_offset, start_ep)
    start_state = start_data["state"][start_frame_idx]  # (10,)

    # Create the real execution environment
    env        = make_env_from_state(start_state)
    start_frame = env.render()
    all_frames  = [start_frame]

    success     = False
    best_dist   = float("inf")
    total_steps = 0
    n_plans     = (MAX_EVAL_STEPS + EXEC_PER_PLAN - 1) // EXEC_PER_PLAN
    warm_mean   = None

    for plan_idx in range(n_plans):
        remaining = MAX_EVAL_STEPS - total_steps
        if remaining <= 0:
            break

        # Current state for planning
        cur_state = np.concatenate([
            env.cue_pos, env.cue_vel,
            env.target_pos, env.target_vel,
            np.zeros(2),
        ])
        cur_dist = min_dist_to_pocket(env.target_pos)
        print(f"\n[plan {plan_idx+1}/{n_plans}]  step {total_steps}/{MAX_EVAL_STEPS}  "
              f"cur_dist_to_pocket = {cur_dist:.2f}")

        plan_actions, plan_best_dist, final_mean = plan_cem_state(cur_state, warm_mean)

        if plan_best_dist < best_dist:
            best_dist = plan_best_dist

        # Warm-start: shift plan by EXEC_PER_PLAN steps
        shift = final_mean[EXEC_PER_PLAN:]
        warm_mean = np.concatenate(
            [shift, np.zeros((EXEC_PER_PLAN, 2), dtype=np.float32)], axis=0
        )

        # Execute first EXEC_PER_PLAN actions in the real env
        exec_slice = plan_actions[:min(EXEC_PER_PLAN, remaining)]
        done = False
        for act in exec_slice:
            obs, _, terminated, truncated, _ = env.step(act)
            all_frames.append(env.render())
            d = min_dist_to_pocket(env.target_pos)
            if d < best_dist:
                best_dist = d
            total_steps += 1
            if terminated:
                success = True
                done    = True
                print(f"    *** TARGET POTTED at step {total_steps}! ***")
                break
            if truncated:
                done = True
                break

        if done:
            break

    env.close()

    print(f"\n[result] Executed {total_steps} total steps  →  "
          f"{'SUCCESS ✓' if success else 'FAILURE ✗'}")
    print(f"         Best dist to pocket = {best_dist:.2f}  "
          f"(pocket radius = {POCKET_RADIUS})")

    save_figure(label, start_frame, all_frames, success, best_dist, out_path)
    return success, best_dist


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Billiards State-Based CEM Evaluation")
    print("  (No world model — simulator-based cost)")
    print("=" * 60)

    f, ep_len, ep_offset = load_dataset()
    results = []

    # Combo A: ep 500 frame 0
    s, d = evaluate(
        f, ep_len, ep_offset,
        label           = "Combo A — ep 500 frame 0 (CEM pots target ball)",
        start_ep        = 500,
        start_frame_idx = 0,
        out_path        = Path(__file__).parent / "eval_state_A.png",
    )
    results.append(("Combo A (ep 500)", s, d))

    # Combo B: ep 100 frame 0
    s, d = evaluate(
        f, ep_len, ep_offset,
        label           = "Combo B — ep 100 frame 0 (CEM pots target ball)",
        start_ep        = 100,
        start_frame_idx = 0,
        out_path        = Path(__file__).parent / "eval_state_B.png",
    )
    results.append(("Combo B (ep 100)", s, d))

    f.close()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, success, dist in results:
        status = "SUCCESS ✓" if success else "FAILURE ✗"
        print(f"  {name:25s}  {status}   best_dist_to_pocket = {dist:.2f}  "
              f"(pocket_radius={POCKET_RADIUS})")
    print("=" * 60)
    print("\n[done] Results saved as eval_state_A.png / eval_state_B.png")


if __name__ == "__main__":
    main()
