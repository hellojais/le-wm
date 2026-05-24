"""probe_auxloss.py — Compare FrameStack vs AuxLoss model representations.

Probes both models at two levels (192-dim ViT CLS token and 32-dim projected
embedding) and prints a 4-column comparison table:

  State dim  | FS-192 | FS-32 | AuxLoss-192 | AuxLoss-32

Key question: did the aux loss recover position R² at 192-dim?
  - FrameStack 192-dim baseline: pos R² ≈ 0.466  (confirmed eviction in encoder)
  - Transformer baseline: pos R² ≈ 0.983
  - Recovery target: AuxLoss 192-dim pos R² > 0.70

Run from le-wm/:
    uv run python experiments/billiards/probe_auxloss.py
"""

import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import torch

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
)
from experiments.billiards.probe_hidden_states import (
    PROBE_EPOCHS,
    N_PROBE_FRAMES,
    PROBE_SEED,
    extract_both_representations,
    train_probe,
)

CKPT_AUXLOSS = Path.home() / ".stable_worldmodel" / "lewm_auxloss_full_best_object.ckpt"


def main():
    print("=" * 70)
    print("  AuxLoss probe — FrameStack vs AuxLoss @ 192-dim and 32-dim")
    print(f"  PROBE_EPOCHS={PROBE_EPOCHS}  N_PROBE_FRAMES={N_PROBE_FRAMES}")
    print("=" * 70)

    print(f"\n[0] Loading dataset: {DATASET_PATH.name}")
    f = h5py.File(DATASET_PATH, "r", swmr=True)
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    print(f"    ✓ {f['pixels'].shape[0]:,} frames")

    print("    Building episode lookup …")
    frame_ep, frame_loc = build_episode_lookup(ep_len, ep_offset)

    results = {}

    for name, ckpt in [("FrameStack", CKPT_FRAMESTACKED),
                        ("AuxLoss",   CKPT_AUXLOSS)]:
        print(f"\n{'─'*70}")
        print(f"  Model: {name}  ({ckpt.name})")
        print(f"{'─'*70}")
        model = load_model(ckpt, name)
        model.eval().requires_grad_(False)

        print(f"\n  Extracting {N_PROBE_FRAMES:,} frames at both levels …")
        hidden, projected, states = extract_both_representations(
            model, f, frame_ep, frame_loc, ep_offset, N_PROBE_FRAMES, seed=PROBE_SEED
        )
        print(f"    ViT CLS (192-dim): {tuple(hidden.shape)}")
        print(f"    Projected (32-dim): {tuple(projected.shape)}")

        state_mean = states.mean(0)
        state_std  = states.std(0).clamp(min=1e-6)
        states_n   = (states - state_mean) / state_std

        r2_192 = train_probe(hidden,    states_n, f"{name} 192-dim (pre-proj)")
        r2_32  = train_probe(projected, states_n, f"{name}  32-dim (post-proj)")

        results[name] = (r2_192, r2_32)

        # free GPU memory before next model
        del model, hidden, projected, states, states_n
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # ── final comparison table ──────────────────────────────────────────
    fs_192,  fs_32  = results["FrameStack"]
    aux_192, aux_32 = results["AuxLoss"]

    print("\n" + "=" * 82)
    print(f"  FINAL COMPARISON  (pos R² target for AuxLoss-192: > 0.70)")
    print("=" * 82)
    hdr = f"  {'State dim':<16}  {'FS-192':>8}  {'FS-32':>8}  {'AuxLoss-192':>12}  {'AuxLoss-32':>10}  {'Δ(192)':>8}"
    print(hdr)
    print("─" * 82)
    for i, name in enumerate(STATE_NAMES):
        tag = "[VEL]" if i in VEL_DIMS else "[POS]" if i in POS_DIMS else "     "
        delta = aux_192[i] - fs_192[i]
        sign  = "+" if delta >= 0 else ""
        print(f"  {tag} {name:<12}  {fs_192[i]:>8.4f}  {fs_32[i]:>8.4f}"
              f"  {aux_192[i]:>12.4f}  {aux_32[i]:>10.4f}"
              f"  {sign}{delta:>7.4f}")
    print("─" * 82)

    print(f"\n  Summary (192-dim CLS token):")
    print(f"    FrameStack:  vel R²={fs_192[list(VEL_DIMS)].mean():.4f}  "
          f"pos R²={fs_192[list(POS_DIMS)].mean():.4f}")
    print(f"    AuxLoss:     vel R²={aux_192[list(VEL_DIMS)].mean():.4f}  "
          f"pos R²={aux_192[list(POS_DIMS)].mean():.4f}")
    pos_delta = aux_192[list(POS_DIMS)].mean() - fs_192[list(POS_DIMS)].mean()
    print(f"    Δ pos R² (AuxLoss - FrameStack): {pos_delta:+.4f}")
    if aux_192[list(POS_DIMS)].mean() > 0.70:
        print("    ✅ RECOVERY: pos R² > 0.70 — aux loss successfully corrected eviction!")
    elif aux_192[list(POS_DIMS)].mean() > fs_192[list(POS_DIMS)].mean() + 0.05:
        print("    ⚠️  PARTIAL: pos R² improved but below 0.70 target.")
    else:
        print("    ❌ NO RECOVERY: pos R² did not improve meaningfully.")

    print(f"\n  Summary (32-dim projected embedding):")
    print(f"    FrameStack:  vel R²={fs_32[list(VEL_DIMS)].mean():.4f}  "
          f"pos R²={fs_32[list(POS_DIMS)].mean():.4f}")
    print(f"    AuxLoss:     vel R²={aux_32[list(VEL_DIMS)].mean():.4f}  "
          f"pos R²={aux_32[list(POS_DIMS)].mean():.4f}")


if __name__ == "__main__":
    main()
