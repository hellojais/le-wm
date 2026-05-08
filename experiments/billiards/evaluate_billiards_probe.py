"""evaluate_billiards_probe.py

Probe-based CEM planning for billiards using the small LeWM model.

Instead of raw embedding L2 distance (which failed because the embedding space
has no task-aligned gradient structure for CEM), this script trains a
lightweight 2-layer MLP probe:

    probe: embedding (32-dim) → state (10-dim)

and uses that decoded state as the CEM cost:

    cost = ||probe(predicted_emb_last) - goal_state_norm||²

This tests whether the embedding *contains* ball position information even
when raw L2 distance doesn't expose it.

Run from le-wm/:
    uv run python evaluate_billiards_probe.py
"""

import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdf5plugin  # must be before h5py
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "billiards-worldmodel"))
from game import BilliardsEnv

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT   = Path.home() / ".stable_worldmodel" / "lewm_small_epoch_8_object.ckpt"
DATASET_PATH = Path.home() / ".stable-wm" / "billiards_expert_train.h5"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cpu")

# Probe training
N_PROBE_FRAMES = 1000   # random frames to use for probe training/test
PROBE_EPOCHS   = 500
PROBE_LR       = 1e-3
PROBE_BATCH    = 128
PROBE_HIDDEN   = 64
STATE_DIM      = 10     # [cue_x, cue_y, cue_vx, cue_vy, tgt_x, tgt_y, tgt_vx, tgt_vy, pkt_x, pkt_y]

# MPC / CEM
HISTORY_SIZE   = 3
PLAN_HORIZON   = 10
EXEC_PER_PLAN  = 3
MAX_EVAL_STEPS = 150
NUM_SAMPLES    = 1000
N_CEM_ITERS    = 20
TOPK           = 50
CEM_CLAMP      = 3.0    # normalised space; dataset std≈1

# ImageNet normalisation (matches train.py preprocessing)
_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

STATE_DIM_NAMES = ["cue_x", "cue_y", "cue_vx", "cue_vy",
                   "tgt_x", "tgt_y", "tgt_vx", "tgt_vy", "pkt_x", "pkt_y"]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame_hwc: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame_hwc.copy()).float() / 255.0
    t = t.permute(2, 0, 1)
    return (t - _MEAN) / _STD


def frames_to_tensor(frames: list) -> torch.Tensor:
    return torch.stack([preprocess_frame(f) for f in frames])


# ─────────────────────────────────────────────────────────────────────────────
# PROBE MODEL — 2-layer MLP
# ─────────────────────────────────────────────────────────────────────────────

class StateProbe(nn.Module):
    """Lightweight MLP: embedding → normalised state vector."""
    def __init__(self, embed_dim: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x):   # x: (..., embed_dim) → (..., state_dim)
        return self.net(x)


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

    # Determine embed_dim by test-encoding a dummy frame
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 3, 96, 96, device=DEVICE)
        emb   = model.encode({"pixels": dummy})["emb"]
    embed_dim = emb.shape[-1]

    n = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Loaded ({n:,} params)  embed_dim={embed_dim}  device={DEVICE}")
    return model, embed_dim


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
    s = ep_offset[ep_idx]
    e = s + ep_len[ep_idx]
    return {
        "pixels": f["pixels"][s:e],
        "state":  f["state"][s:e],
        "action": f["action"][s:e],
    }


def action_stats(f):
    acts = f["action"][:]
    mean = acts.mean(0).astype(np.float32)
    std  = acts.std(0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    print(f"     action mean  = {mean}")
    print(f"     action std   = {std}")
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN PROBE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_probe_data(model, f, n_frames: int, seed: int = 42):
    """Sample n_frames randomly from dataset; return (embeddings, states) tensors."""
    rng     = np.random.default_rng(seed)
    n_total = f["pixels"].shape[0]
    indices = np.sort(rng.choice(n_total, size=n_frames, replace=False))

    print(f"    Encoding {n_frames} frames ...", flush=True)
    emb_list   = []
    state_list = []
    batch_size = 32

    for start in range(0, n_frames, batch_size):
        idx_batch = indices[start : start + batch_size]
        frames_np = f["pixels"][idx_batch]       # (B, H, W, C) uint8
        states_np = f["state"][idx_batch]        # (B, 10) float32

        frames_t = torch.stack([preprocess_frame(frames_np[j])
                                 for j in range(len(idx_batch))])   # (B, C, H, W)
        frames_t = frames_t.unsqueeze(1).to(DEVICE)                 # (B, 1, C, H, W)

        emb = model.encode({"pixels": frames_t})["emb"]             # (B, 1, D)
        emb_list.append(emb[:, 0, :].cpu())
        state_list.append(torch.from_numpy(states_np.astype(np.float32)))

        pct = min(100, int((start + batch_size) / n_frames * 100))
        print(f"      {pct:3d}%", end="\r", flush=True)

    print()
    embs   = torch.cat(emb_list,   dim=0)   # (N, D)
    states = torch.cat(state_list, dim=0)   # (N, 10)
    return embs, states


def train_probe(model, f, embed_dim: int):
    """
    Train StateProbe on embedding → state.

    Returns:
        probe       — trained StateProbe on DEVICE, eval mode, no grad
        state_mean  — (10,) CPU tensor, mean of training states (raw units)
        state_std   — (10,) CPU tensor, std  of training states (raw units)
    """
    print(f"\n[3] Training state probe  (embed_dim={embed_dim} → state_dim={STATE_DIM})")
    embs, states = extract_probe_data(model, f, N_PROBE_FRAMES)

    # Normalise state (positions ~0–512, velocities ~small; need common scale)
    state_mean = states.mean(0)                         # (10,) CPU
    state_std  = states.std(0).clamp(min=1e-6)          # (10,) CPU
    states_norm = (states - state_mean) / state_std     # (N, 10) CPU

    probe    = StateProbe(embed_dim, STATE_DIM, PROBE_HIDDEN).to(DEVICE)
    opt      = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    embs_d   = embs.to(DEVICE)
    states_d = states_norm.to(DEVICE)
    n        = len(embs_d)

    best_loss = float("inf")
    for epoch in range(PROBE_EPOCHS):
        perm       = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        nb         = 0
        for bi in range(0, n, PROBE_BATCH):
            idx  = perm[bi : bi + PROBE_BATCH]
            pred = probe(embs_d[idx])
            loss = F.mse_loss(pred, states_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb         += 1
        epoch_loss /= nb
        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if epoch % 100 == 0 or epoch == PROBE_EPOCHS - 1:
            with torch.no_grad():
                pred_all = probe(embs_d)
            ss_res    = (pred_all - states_d).pow(2).sum(0)
            ss_tot    = (states_d - states_d.mean(0)).pow(2).sum(0).clamp(min=1e-8)
            r2_dims   = (1 - ss_res / ss_tot).cpu().numpy()
            mean_r2   = r2_dims.mean()
            print(f"    epoch {epoch:4d}/{PROBE_EPOCHS}  "
                  f"MSE={epoch_loss:.5f}  mean_R²={mean_r2:.4f}")

    # Final per-dimension R² report
    with torch.no_grad():
        pred_all = probe(embs_d)
    ss_res  = (pred_all - states_d).pow(2).sum(0)
    ss_tot  = (states_d - states_d.mean(0)).pow(2).sum(0).clamp(min=1e-8)
    r2_dims = (1 - ss_res / ss_tot).cpu().numpy()
    print(f"\n    ✓ Probe R² per dimension:")
    for name, r in zip(STATE_DIM_NAMES, r2_dims):
        bar = "█" * max(0, int(r * 20))
        print(f"      {name:8s}: {r:+.4f}  {bar}")
    print(f"    Probe mean R² = {r2_dims.mean():.4f}   "
          f"best MSE = {best_loss:.5f}")

    probe.eval()
    probe.requires_grad_(False)
    return probe, state_mean, state_std


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CEM PLANNING WITH PROBE COST
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def plan_cem_probe(model, probe, embed_dim: int,
                   ctx_frames: list,
                   goal_state_norm: torch.Tensor,   # (state_dim,) on DEVICE
                   act_mean_np: np.ndarray, act_std_np: np.ndarray,
                   hist_act_norm: np.ndarray | None = None,
                   warm_mean_norm: np.ndarray | None = None):
    """
    CEM planning with probe-decoded state as cost function.

    cost = ||probe(predicted_emb_last) - goal_state_norm||²

    Args:
        goal_state_norm: (state_dim,) normalised goal state on DEVICE
    Returns:
        best_actions_env:  (PLAN_HORIZON, 2) in env units
        best_cost:         scalar
        cem_mean_norm:     (PLAN_HORIZON, 2) normalised CEM mean for warm-start
    """
    B, S, H, act_dim = 1, NUM_SAMPLES, HISTORY_SIZE, 2

    ctx     = frames_to_tensor(ctx_frames).to(DEVICE)                          # (H, C, h, w)
    ctx_bst = ctx.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1, -1)      # (B, S, H, C, h, w)

    if hist_act_norm is not None:
        ha       = torch.from_numpy(hist_act_norm.astype(np.float32)).to(DEVICE)
        hist_act = ha.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)           # (B, S, H, 2)
    else:
        hist_act = torch.zeros(B, S, H, act_dim, device=DEVICE)

    # goal_state_norm: (state_dim,) → (B, S, state_dim) for broadcasting
    goal_bs = goal_state_norm.unsqueeze(0).unsqueeze(0).expand(B, S, -1)       # (B, S, state_dim)

    if warm_mean_norm is not None:
        cem_mean = torch.from_numpy(warm_mean_norm.astype(np.float32)).to(DEVICE)
        cem_std  = torch.ones(PLAN_HORIZON, act_dim, device=DEVICE) * 0.5
    else:
        cem_mean = torch.zeros(PLAN_HORIZON, act_dim, device=DEVICE)
        cem_std  = torch.ones( PLAN_HORIZON, act_dim, device=DEVICE)

    best_cost        = float("inf")
    best_actions_env = np.zeros((PLAN_HORIZON, act_dim), dtype=np.float32)

    print(f"    CEM: {N_CEM_ITERS} iters × {S} samples × horizon={PLAN_HORIZON}  "
          f"clamp=±{CEM_CLAMP}  cost=probe_state_L2²")

    for it in range(N_CEM_ITERS):
        noise   = torch.randn(S, PLAN_HORIZON, act_dim, device=DEVICE)
        fut_act = (cem_mean + cem_std * noise).clamp(-CEM_CLAMP, CEM_CLAMP)   # (S, T, 2)

        action_seq = torch.cat([hist_act, fut_act.unsqueeze(0)], dim=2)        # (B, S, H+T, 2)

        info = {"pixels": ctx_bst}
        model.rollout(info, action_seq, history_size=HISTORY_SIZE)

        pred_emb  = info["predicted_emb"]                                      # (B, S, H+T+1, D)
        last_emb  = pred_emb[:, :, -1, :]                                     # (B, S, D)

        # Decode state from predicted embedding via probe
        pred_state_flat = probe(last_emb.reshape(-1, embed_dim))              # (B*S, state_dim)
        pred_state      = pred_state_flat.view(B, S, -1)                      # (B, S, state_dim)

        cost    = (pred_state - goal_bs).pow(2).sum(-1)                       # (B, S)
        costs_s = cost[0]                                                      # (S,)

        _, top_idx = torch.topk(-costs_s, k=TOPK)
        elites   = fut_act[top_idx]
        cem_mean = elites.mean(0)
        cem_std  = elites.std(0).clamp(min=1e-3)

        it_best      = costs_s.min().item()
        mean_act_mag = elites.abs().mean().item()
        print(f"      iter {it+1:2d}/{N_CEM_ITERS}  best_cost = {it_best:.4f}  "
              f"elite_act_mag = {mean_act_mag:.3f}")

        if it_best < best_cost:
            best_cost        = it_best
            best_norm_np     = cem_mean.cpu().numpy()
            best_actions_env = best_norm_np * act_std_np + act_mean_np

    print(f"    ✓ Planning done  best_cost = {best_cost:.4f}  "
          f"best_act_mag (env) = {np.abs(best_actions_env).mean():.3f}")
    return best_actions_env, best_cost, cem_mean.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MPC EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, probe, embed_dim: int,
             state_mean: torch.Tensor, state_std: torch.Tensor,
             f, ep_len, ep_offset, act_mean, act_std,
             label, start_ep, start_frame_idx, goal_ep, out_path):
    """
    Receding-horizon MPC with probe-decoded state cost.

    state_mean, state_std: (10,) CPU tensors.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Start: ep {start_ep}  frame {start_frame_idx}")
    print(f"  Goal : ep {goal_ep}   last frame  (HDF5 ground-truth state as target)")
    print(f"  MPC: plan={PLAN_HORIZON} exec={EXEC_PER_PLAN}/plan budget={MAX_EVAL_STEPS}")
    print(f"{'─' * 60}")

    start_data = get_episode(f, ep_len, ep_offset, start_ep)
    goal_data  = get_episode(f, ep_len, ep_offset, goal_ep)

    ctx_indices = [max(0, start_frame_idx - HISTORY_SIZE + 1 + i)
                   for i in range(HISTORY_SIZE)]
    ctx_frames  = [start_data["pixels"][i] for i in ctx_indices]
    start_state = start_data["state"][start_frame_idx]

    goal_idx       = max(0, len(goal_data["pixels"]) - 4)
    goal_frame_img = goal_data["pixels"][goal_idx]
    goal_state_raw = torch.from_numpy(goal_data["state"][goal_idx].astype(np.float32))  # CPU

    # Normalise goal state → DEVICE
    goal_state_norm = ((goal_state_raw - state_mean) / state_std).to(DEVICE)

    print(f"  Goal state (raw):  {goal_state_raw.numpy().round(1)}")
    print(f"  Goal target pos:   ({goal_state_raw[4]:.1f}, {goal_state_raw[5]:.1f})  "
          f"[should be near pocket at corners ~20 or ~492]")

    # Log probe quality at start
    with torch.no_grad():
        start_img = preprocess_frame(ctx_frames[-1]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        start_emb = model.encode({"pixels": start_img})["emb"][0, 0]
        start_pred_norm = probe(start_emb.unsqueeze(0))[0]
        start_pred_raw  = start_pred_norm * state_std.to(DEVICE) + state_mean.to(DEVICE)
    print(f"  Start state (real): {start_state.round(1)}")
    print(f"  Start state (pred): {start_pred_raw.cpu().numpy().round(1)}")

    # Initialise environment
    env = BilliardsEnv(render_mode="rgb_array")
    env.reset()
    env.cue_pos    = start_state[0:2].astype(np.float32).copy()
    env.cue_vel    = start_state[2:4].astype(np.float32).copy()
    env.target_pos = start_state[4:6].astype(np.float32).copy()
    env.target_vel = start_state[6:8].astype(np.float32).copy()
    env._step_count    = 0
    env._target_potted = False

    all_frames       = [env.render()]
    exec_norm_buffer = []
    cem_warm_mean    = None
    success          = False
    best_cost        = float("inf")
    total_steps      = 0
    n_plans          = (MAX_EVAL_STEPS + EXEC_PER_PLAN - 1) // EXEC_PER_PLAN
    act_dim          = 2

    for plan_idx in range(n_plans):
        remaining = MAX_EVAL_STEPS - total_steps
        if remaining <= 0:
            break

        # History action context
        if len(exec_norm_buffer) > 0:
            h = np.array(exec_norm_buffer[-HISTORY_SIZE:])
            if len(h) < HISTORY_SIZE:
                h = np.pad(h, ((HISTORY_SIZE - len(h), 0), (0, 0)))
            hist_arr = h.astype(np.float32)
        else:
            hist_arr = None

        # Probe-decoded current state distance to goal (for monitoring)
        with torch.no_grad():
            cur_img  = preprocess_frame(ctx_frames[-1]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            cur_emb  = model.encode({"pixels": cur_img})["emb"][0, 0]
            cur_norm = probe(cur_emb.unsqueeze(0))[0]
            cur_dist = (cur_norm - goal_state_norm).pow(2).sum().sqrt().item()

        print(f"\n[P] CEM plan {plan_idx+1}/{n_plans}  "
              f"(step {total_steps}/{MAX_EVAL_STEPS})  "
              f"probe_dist={cur_dist:.3f} ...")

        plan_actions, cost, plan_norm = plan_cem_probe(
            model, probe, embed_dim,
            ctx_frames, goal_state_norm, act_mean, act_std,
            hist_act_norm  = hist_arr,
            warm_mean_norm = cem_warm_mean,
        )
        best_cost = min(best_cost, cost)

        # Warm-start: shift plan forward by EXEC_PER_PLAN steps
        shift         = plan_norm[EXEC_PER_PLAN:]
        cem_warm_mean = np.concatenate(
            [shift, np.zeros((PLAN_HORIZON - len(shift), act_dim), dtype=np.float32)], axis=0
        )

        exec_slice = plan_actions[:min(EXEC_PER_PLAN, remaining)]
        done = False
        for act in exec_slice:
            _, _, terminated, truncated, _ = env.step(act)
            all_frames.append(env.render())
            exec_norm_buffer.append((act - act_mean) / np.maximum(act_std, 1e-6))
            total_steps += 1
            if terminated:
                success = True
                done    = True
                break
            if truncated:
                done = True
                break

        ctx_frames = [all_frames[max(0, len(all_frames) - HISTORY_SIZE + i)]
                      for i in range(HISTORY_SIZE)]
        if done:
            break

    env.close()
    print(f"\n[4] Executed {total_steps} total steps  →  "
          f"{'SUCCESS ✓' if success else 'FAILURE ✗'}")

    print(f"\n[5] Saving figure ...")
    save_figure(label, start_data["pixels"][start_frame_idx], goal_frame_img,
                all_frames, success, best_cost, out_path)
    return success, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(label, start_frame, goal_frame, exec_frames, success, cost, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    status = "SUCCESS ✓" if success else "FAILURE ✗"
    color  = "green" if success else "crimson"
    fig.suptitle(
        f"{label}\n{status}   |   Best probe-state cost: {cost:.4f}",
        fontsize=12, fontweight="bold", color=color,
    )
    axes[0].imshow(start_frame);   axes[0].set_title("Start");   axes[0].axis("off")
    axes[1].imshow(goal_frame);    axes[1].set_title("Goal");    axes[1].axis("off")
    n   = len(exec_frames)
    mid = exec_frames[n // 2]
    strip = np.concatenate([exec_frames[0], mid, exec_frames[-1]], axis=1)
    axes[2].imshow(strip)
    axes[2].set_title(f"Execution: start | mid | end  ({n} steps)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LeWM Billiards — Probe-based CEM Planning")
    print(f"  Checkpoint  : {CHECKPOINT.name}")
    print(f"  Device      : {DEVICE}")
    print(f"  Probe       : Linear({STATE_DIM}←{PROBE_HIDDEN}←embed_dim)  "
          f"trained on {N_PROBE_FRAMES} frames")
    print("=" * 60)

    model, embed_dim     = load_model()
    f, ep_len, ep_offset = load_dataset()

    print("\n[2b] Action normalisation stats ...")
    act_mean, act_std = action_stats(f)

    probe, state_mean, state_std = train_probe(model, f, embed_dim)

    results = []

    s, c = evaluate(
        model, probe, embed_dim, state_mean, state_std,
        f, ep_len, ep_offset, act_mean, act_std,
        label           = "Combo A — Same episode (ep 500 start → ep 500 goal)",
        start_ep        = 500,
        start_frame_idx = 0,
        goal_ep         = 500,
        out_path        = Path(__file__).parent / "probe_result_A.png",
    )
    results.append(("Combo A (same ep)", s, c))

    s, c = evaluate(
        model, probe, embed_dim, state_mean, state_std,
        f, ep_len, ep_offset, act_mean, act_std,
        label           = "Combo B — Novel cross-episode (ep 100 start → ep 3000 goal)",
        start_ep        = 100,
        start_frame_idx = 0,
        goal_ep         = 3000,
        out_path        = Path(__file__).parent / "probe_result_B.png",
    )
    results.append(("Combo B (cross-ep)", s, c))

    f.close()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, success, cost in results:
        status = "SUCCESS ✓" if success else "FAILURE ✗"
        print(f"  {name:30s}  {status}   best probe-state cost = {cost:.6f}")
    print("=" * 60)
    print("\n[6] Done.  Results saved as: probe_result_A.png  probe_result_B.png")


if __name__ == "__main__":
    main()
