"""
Training curves for LeWM billiards experiments.
Plots validation prediction loss for original (embed_dim=192) and
small (embed_dim=32) models on a shared log-scale figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "training_curves.png"

# ── hardcoded training data ──────────────────────────────────────────────────
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

orig_best_epoch  = int(np.argmin(orig_val_pred_loss))  + 1   # epoch 20 (0.007)
small_best_epoch = int(np.argmin(small_val_pred_loss)) + 1   # epoch 8  (0.0028)
orig_best_val    = min(orig_val_pred_loss)
small_best_val   = min(small_val_pred_loss)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    orig_epochs, orig_val_pred_loss,
    color="steelblue", linewidth=2, marker="o", markersize=4,
    label="Original model (192 dims, λ=0.09, 20 epochs)",
)
ax.plot(
    small_epochs, small_val_pred_loss,
    color="crimson", linewidth=2, marker="o", markersize=4,
    label="Small model (32 dims, λ=0.01, 10 epochs)",
)

# ── best-epoch markers ────────────────────────────────────────────────────────
ax.plot(
    orig_best_epoch, orig_best_val,
    marker="*", color="steelblue", markersize=16, zorder=5,
    label=f"Original best  (epoch {orig_best_epoch}, loss={orig_best_val:.4f})",
)
ax.plot(
    small_best_epoch, small_best_val,
    marker="*", color="crimson", markersize=16, zorder=5,
    label=f"Small best  (epoch {small_best_epoch}, loss={small_best_val:.4f})",
)

# ── dashed vertical lines at best epochs ─────────────────────────────────────
ax.axvline(orig_best_epoch,  color="steelblue", linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(small_best_epoch, color="crimson",   linestyle="--", linewidth=1, alpha=0.5)

# ── "2.5× better" annotation ─────────────────────────────────────────────────
ratio = orig_best_val / small_best_val
# arrow from orig best point to small best point (horizontally at orig_best_epoch)
mid_x = small_best_epoch + 0.3
ax.annotate(
    f"{ratio:.1f}× better",
    xy=(small_best_epoch, small_best_val),
    xytext=(small_best_epoch + 1.2, small_best_val * 2.5),
    fontsize=11, fontweight="bold", color="black",
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
)

# ── axes formatting ───────────────────────────────────────────────────────────
ax.set_yscale("log")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Validation Prediction Loss (log scale)", fontsize=13)
ax.set_title("Training Curves: LeWM Billiards", fontsize=15, fontweight="bold")
ax.set_xticks(range(1, max(len(orig_val_pred_loss), len(small_val_pred_loss)) + 1))
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
print(f"Original best:  epoch {orig_best_epoch}, loss={orig_best_val:.4f}")
print(f"Small best:     epoch {small_best_epoch}, loss={small_best_val:.4f}")
print(f"Improvement:    {ratio:.2f}×")
