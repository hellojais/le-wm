"""visualize_planning.py

Runs CEM billiards planning for two starting episodes and saves:
  combo_a_planning.gif  — animated 3-panel execution for ep 500
  combo_b_planning.gif  — animated 3-panel execution for ep 100
  planning_results.png  — static 2×3 comparison figure (start/goal/final)

Each GIF shows:
  Left panel  : Start frame (static)
  Middle panel: Goal frame = last frame of the episode (ball potted, static)
  Right panel : Animated CEM execution, step counter, success badge

Run from the le-wm directory:
    uv run python visualize_planning.py
"""

import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import hdf5plugin  # must precede h5py
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
from game import BilliardsEnv, POCKET_POSITIONS, POCKET_RADIUS

# ── CEM / MPC parameters (must match evaluate_billiards_state.py) ─────────────
DATASET_PATH   = Path.home() / ".stable-wm" / "billiards_expert_train.h5"
PLAN_HORIZON   = 20
EXEC_PER_PLAN  = 5
MAX_EVAL_STEPS = 300
NUM_SAMPLES    = 500
N_CEM_ITERS    = 20
TOPK           = 50
ACT_LO         = -30.0
ACT_HI         =  30.0

# ── GIF rendering parameters ─────────────────────────────────────────────────
GIF_FPS      = 12    # playback speed
HOLD_START   = 24    # frames to hold on start panel before execution (2 s)
HOLD_GOAL    = 30    # frames to hold while "Planning…" label is shown (2.5 s)
EXEC_REPEAT  = 4     # each execution frame repeated N times  (~3 fps effective)
HOLD_END     = 48    # frames to hold on the final result (4 s)
GIF_DPI      = 90    # resolution; keep low for reasonable file size


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def min_dist_to_pocket(target_pos: np.ndarray) -> float:
    return float(np.linalg.norm(POCKET_POSITIONS - target_pos, axis=1).min())


def make_env_from_state(state: np.ndarray) -> BilliardsEnv:
    env = BilliardsEnv(render_mode="rgb_array")
    env.reset()
    env.cue_pos        = state[0:2].astype(np.float32).copy()
    env.cue_vel        = state[2:4].astype(np.float32).copy()
    env.target_pos     = state[4:6].astype(np.float32).copy()
    env.target_vel     = state[6:8].astype(np.float32).copy()
    env._step_count    = 0
    env._target_potted = False
    return env


def simulate_sequence(start_state: np.ndarray, actions: np.ndarray) -> float:
    """Simulate action sequence, return min dist to pocket (no pygame surfaces)."""
    env = make_env_from_state(start_state)
    best = min_dist_to_pocket(env.target_pos)
    for act in actions:
        _, _, terminated, truncated, _ = env.step(act)
        d = min_dist_to_pocket(env.target_pos)
        if d < best:
            best = d
        if terminated or truncated:
            break
    # env.close() is a no-op when render was never called (no pygame surface)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# CEM PLANNER
# ─────────────────────────────────────────────────────────────────────────────

def plan_cem(start_state: np.ndarray,
             warm_mean: np.ndarray | None = None,
             verbose: bool = True) -> tuple:
    """
    Cross-Entropy Method over simulator rollouts.

    Returns:
        best_actions : (PLAN_HORIZON, 2)
        best_cost    : float — min dist to pocket of best sequence
        final_mean   : (PLAN_HORIZON, 2) — for warm-starting the next call
    """
    act_dim = 2
    if warm_mean is not None:
        cem_mean = warm_mean.copy()
        cem_std  = np.full((PLAN_HORIZON, act_dim), 5.0, dtype=np.float32)
    else:
        cem_mean = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)
        cem_std  = np.full((PLAN_HORIZON, act_dim), 15.0, dtype=np.float32)

    best_cost    = float("inf")
    best_actions = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)

    for it in range(N_CEM_ITERS):
        noise   = np.random.randn(NUM_SAMPLES, PLAN_HORIZON, act_dim).astype(np.float32)
        samples = (cem_mean[None] + cem_std[None] * noise).clip(ACT_LO, ACT_HI)
        costs   = np.array([simulate_sequence(start_state, samples[i])
                            for i in range(NUM_SAMPLES)], dtype=np.float32)

        elite_idx = np.argpartition(costs, TOPK)[:TOPK]
        elites    = samples[elite_idx]
        cem_mean  = elites.mean(0)
        cem_std   = elites.std(0).clip(min=0.5)

        it_best = float(costs[elite_idx].min())
        if verbose:
            print(f"      iter {it+1:2d}/{N_CEM_ITERS}  best_dist = {it_best:.2f}")

        if it_best < best_cost:
            best_cost    = it_best
            best_actions = samples[elite_idx[costs[elite_idx].argmin()]].copy()

    return best_actions, best_cost, cem_mean


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING RUNNER  (collects frames for visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def run_planning(f, ep_len, ep_offset,
                 start_ep: int, start_frame_idx: int, label: str) -> dict:
    """
    Run full MPC planning loop and record every rendered frame.

    Returns a dict with keys:
        frames      : list of HWC uint8 arrays (start frame + each execution step)
        step_labels : list of str, one per frame
        success     : bool
        best_dist   : float
        total_steps : int
        start_frame : HWC uint8 — first frame
    """
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  MPC: plan={PLAN_HORIZON}, exec={EXEC_PER_PLAN}/plan, "
          f"budget={MAX_EVAL_STEPS}")
    print(f"{'═' * 60}")

    s = int(ep_offset[start_ep])
    start_state = f["state"][s + start_frame_idx]

    env         = make_env_from_state(start_state)
    start_frame = env.render()
    all_frames  = [start_frame]
    step_labels = ["Step 0  (start)"]

    success     = False
    best_dist   = float("inf")
    total_steps = 0
    warm_mean   = None
    n_plans     = (MAX_EVAL_STEPS + EXEC_PER_PLAN - 1) // EXEC_PER_PLAN

    for plan_idx in range(n_plans):
        remaining = MAX_EVAL_STEPS - total_steps
        if remaining <= 0:
            break

        cur_state = np.concatenate([
            env.cue_pos, env.cue_vel,
            env.target_pos, env.target_vel,
            np.zeros(2),
        ])
        cur_dist = min_dist_to_pocket(env.target_pos)
        print(f"\n  [plan {plan_idx + 1}/{n_plans}]  "
              f"step {total_steps}/{MAX_EVAL_STEPS}  "
              f"dist_to_pocket = {cur_dist:.1f}")

        plan_actions, plan_best, final_mean = plan_cem(cur_state, warm_mean)
        if plan_best < best_dist:
            best_dist = plan_best

        # Warm-start: shift plan forward by EXEC_PER_PLAN steps
        shift     = final_mean[EXEC_PER_PLAN:]
        warm_mean = np.concatenate(
            [shift, np.zeros((EXEC_PER_PLAN, 2), dtype=np.float32)], axis=0
        )

        exec_slice = plan_actions[:min(EXEC_PER_PLAN, remaining)]
        done = False
        for act in exec_slice:
            _, _, terminated, truncated, _ = env.step(act)
            total_steps += 1
            frame = env.render()
            all_frames.append(frame)

            d = min_dist_to_pocket(env.target_pos)
            if d < best_dist:
                best_dist = d

            lbl = f"Step {total_steps}"
            if terminated:
                lbl += "  ✓ POTTED!"
                success = True
                done    = True
                all_frames.append(frame)   # extra copy so hold_end feels natural
                step_labels.append(lbl)
                break
            step_labels.append(lbl)
            if truncated:
                done = True
                break

        if done:
            break

    env.close()

    print(f"\n  → {'SUCCESS ✓' if success else 'FAILURE ✗'}"
          f"  in {total_steps} steps  |  "
          f"best dist to pocket = {best_dist:.2f}  (pocket_radius={POCKET_RADIUS})")

    return {
        "frames":      all_frames,
        "step_labels": step_labels,
        "success":     success,
        "best_dist":   best_dist,
        "total_steps": total_steps,
        "start_frame": start_frame,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATASET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    print(f"\n[dataset] Loading {DATASET_PATH.name} …")
    f         = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"          ✓ {len(ep_len)} episodes, {f['pixels'].shape[0]:,} frames")
    return f, ep_len, ep_offset


def get_goal_frame(f, ep_len, ep_offset, ep_idx: int) -> np.ndarray:
    """Return the last frame of an episode (ball is already in a pocket)."""
    s = int(ep_offset[ep_idx])
    e = s + int(ep_len[ep_idx])
    return f["pixels"][e - 1]


# ─────────────────────────────────────────────────────────────────────────────
# GIF CREATION
# ─────────────────────────────────────────────────────────────────────────────

def make_gif(result: dict, goal_frame: np.ndarray,
             combo_label: str, out_path: Path) -> None:
    """
    Save a 3-panel animated GIF.

    Sequence:
      Phase 1 HOLD_START  — right panel shows start frame; "Start" label
      Phase 2 HOLD_GOAL   — right panel shows start frame; "Planning…" label
      Phase 3 execution   — right panel animates through execution frames
      Phase 4 HOLD_END    — right panel holds on final frame; "SUCCESS!" label
    """
    exec_frames  = result["frames"]
    step_labels  = result["step_labels"]
    total_steps  = result["total_steps"]
    best_dist    = result["best_dist"]
    success      = result["success"]

    # ── Build frame sequence: list of (frame, label_str) ─────────────────────
    seq: list[tuple] = []
    seq += [(exec_frames[0],  "Start")]                  * HOLD_START
    seq += [(exec_frames[0],  "Planning…")]              * HOLD_GOAL

    for frame, lbl in zip(exec_frames, step_labels):
        seq += [(frame, lbl)] * EXEC_REPEAT

    final_lbl = (f"Step {total_steps}  ✓ SUCCESS!"
                 if success else f"Step {total_steps}  ✗ FAILURE")
    seq += [(exec_frames[-1], final_lbl)]                * HOLD_END

    # ── Figure / axes ─────────────────────────────────────────────────────────
    BG   = "#111827"
    FG   = "#e2e8f0"
    ACT  = "#34d399" if success else "#f87171"   # green / red
    GOLD = "#fbbf24"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw={"wspace": 0.06})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Static heading
    fig.text(0.5, 0.97, combo_label,
             ha="center", va="top", fontsize=13, fontweight="bold", color=FG)
    fig.text(0.5, 0.92,
             f"CEM planning — simulator-based cost  |  pocket radius = {POCKET_RADIUS} px",
             ha="center", va="top", fontsize=9.5, color="#94a3b8")

    # Panel titles
    for ax, title, color in zip(
        axes,
        ["Start", "Goal  (episode end)", "Execution"],
        ["#93c5fd", "#fde68a", ACT],
    ):
        ax.set_title(title, color=color, fontsize=11, pad=5)
        ax.axis("off")

    # Left: start (always static)
    axes[0].imshow(result["start_frame"], interpolation="nearest")
    _draw_frame_border(axes[0], "#93c5fd")

    # Middle: goal (always static)
    axes[1].imshow(goal_frame, interpolation="nearest")
    _draw_frame_border(axes[1], "#fde68a")

    # Right: animated
    im_exec  = axes[2].imshow(exec_frames[0], interpolation="nearest")
    step_txt = axes[2].text(
        0.5, -0.04, "Start", transform=axes[2].transAxes,
        ha="center", va="top", fontsize=10.5, color=FG,
    )
    badge = axes[2].text(
        0.5, -0.13, "",
        transform=axes[2].transAxes,
        ha="center", va="top", fontsize=10, color=ACT, fontweight="bold",
    )
    _draw_frame_border(axes[2], ACT)

    plt.subplots_adjust(top=0.88, bottom=0.12)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame_idx: int):
        frame, lbl = seq[frame_idx]
        im_exec.set_data(frame)
        step_txt.set_text(lbl)

        if "SUCCESS" in lbl or "POTTED" in lbl:
            step_txt.set_color(ACT)
            badge.set_text(
                f"✓ Potted in {total_steps} steps   "
                f"|   best dist = {best_dist:.1f} px"
            )
        elif "Planning" in lbl:
            step_txt.set_color(GOLD)
            badge.set_text("")
        else:
            step_txt.set_color(FG)
            badge.set_text("")

        return [im_exec, step_txt, badge]

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=len(seq), interval=1000 / GIF_FPS, blit=True
    )

    print(f"  Saving {len(seq)}-frame GIF @ {GIF_FPS} fps → {out_path}")
    writer = mpl_animation.PillowWriter(fps=GIF_FPS)
    ani.save(str(out_path), writer=writer, dpi=GIF_DPI)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


def _draw_frame_border(ax, color: str, lw: float = 2.5):
    """Draw a coloured rectangle border around an axes panel."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def make_static_figure(result_a: dict, result_b: dict,
                       goal_a: np.ndarray, goal_b: np.ndarray,
                       out_path: Path) -> None:
    """
    2-row × 3-column figure.
      Row  : Combo A (top) / Combo B (bottom)
      Col  : Start | Goal (episode end) | Final execution frame
    """
    BG   = "#111827"
    FG   = "#e2e8f0"
    SB   = "#94a3b8"

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.05},
    )
    fig.patch.set_facecolor(BG)

    col_colors  = ["#93c5fd", "#fde68a", "#34d399"]
    col_titles  = ["Start frame", "Goal  (episode end)", "Final execution frame"]

    combos = [
        ("Combo A — Episode 500  (same-episode)",  result_a, goal_a),
        ("Combo B — Episode 100  (novel start)",   result_b, goal_b),
    ]

    for row, (combo_label, result, goal) in enumerate(combos):
        success = result["success"]
        s_color = "#34d399" if success else "#f87171"
        status  = "SUCCESS ✓" if success else "FAILURE ✗"

        # Row annotation on the left
        fig.text(
            0.005, 0.73 - row * 0.46,
            f"{combo_label}\n"
            f"{status}  |  {result['total_steps']} steps  "
            f"|  best dist = {result['best_dist']:.1f} px",
            va="center", fontsize=9.5, color=s_color,
            rotation=90, ha="center",
        )

        panels = [result["start_frame"], goal, result["frames"][-1]]
        for col, (img, title, tc) in enumerate(
                zip(panels, col_titles, col_colors)):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            _draw_frame_border(ax, tc, lw=2.0)
            if row == 0:
                ax.set_title(title, color=tc, fontsize=10.5, pad=5)

    fig.suptitle(
        "Billiards CEM Planning — State-Based Cost\n"
        "CEM optimises action sequences via simulator rollouts; "
        "cost = min distance from target ball to any pocket",
        fontsize=12, fontweight="bold", color=FG, y=1.01,
    )

    # Legend
    handles = [
        mpatches.Patch(color="#93c5fd", label="Start frame"),
        mpatches.Patch(color="#fde68a", label="Goal (episode end — ball potted)"),
        mpatches.Patch(color="#34d399", label="Final execution frame"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               facecolor=BG, edgecolor=SB, labelcolor=FG,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved static figure: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Billiards CEM — Visualization Script")
    print("=" * 60)

    f, ep_len, ep_offset = load_dataset()

    result_a = run_planning(
        f, ep_len, ep_offset,
        start_ep=500, start_frame_idx=0,
        label="Combo A — Episode 500, frame 0",
    )
    result_b = run_planning(
        f, ep_len, ep_offset,
        start_ep=100, start_frame_idx=0,
        label="Combo B — Episode 100, frame 0",
    )

    goal_a = get_goal_frame(f, ep_len, ep_offset, ep_idx=500)
    goal_b = get_goal_frame(f, ep_len, ep_offset, ep_idx=100)

    f.close()

    out_dir = Path(__file__).parent
    print("\n[output] Generating GIFs and static figure …")

    make_gif(
        result_a, goal_a,
        combo_label="Combo A — Episode 500  |  CEM billiards planning",
        out_path=out_dir / "combo_a_planning.gif",
    )
    make_gif(
        result_b, goal_b,
        combo_label="Combo B — Episode 100  |  CEM billiards planning",
        out_path=out_dir / "combo_b_planning.gif",
    )
    make_static_figure(
        result_a, result_b, goal_a, goal_b,
        out_path=out_dir / "planning_results.png",
    )

    print("\n" + "=" * 60)
    print("  Done!  Files written:")
    print("    combo_a_planning.gif")
    print("    combo_b_planning.gif")
    print("    planning_results.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
