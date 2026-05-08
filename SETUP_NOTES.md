# LeWM × Billiards — Setup Notes (Apple M5 Max)

## 1. Install Commands (M5 Max / MPS)

```bash
# Clone and enter the repo
cd ~/your-projects-folder

# Create a venv for le-wm
python3 -m venv le-wm/.venv
source le-wm/.venv/bin/activate

# Install PyTorch with MPS support (ships in the default macOS wheel)
pip install torch torchvision torchaudio

# Install stable-worldmodel in editable mode (local dependency)
pip install -e stable-worldmodel

# Install le-wm dependencies
pip install -e le-wm

# Verify MPS is available
python - <<'PY'
import torch
print("MPS available :", torch.backends.mps.is_available())
print("MPS built     :", torch.backends.mps.is_built())
t = torch.ones(3, device="mps")
print("Tensor on MPS :", t)
PY
```

---

## 2. loss.py Fix Applied

**File:** `stable-worldmodel/stable_worldmodel/wm/loss.py`  
**Line:** 30 (inside `SIGReg.forward`)

| Before | After |
|--------|-------|
| `A = torch.randn(..., device='cuda')` | `A = torch.randn(..., device=proj.device)` |

**Why:** The hardcoded `device='cuda'` caused a crash on Apple Silicon (MPS)
and CPU-only machines. Using `proj.device` ensures the random projection
tensor is always allocated on the same device as the input.

---

## 3. Launch Billiards Training

Make sure your dataset (`billiards_expert_train.h5`) is in the stable-worldmodel
cache directory (or symlinked there), then run:

```bash
cd ~/your-projects-folder/le-wm
source .venv/bin/activate

# Train with billiards data config (MPS auto-detected by train.py)
# NOTE: patch_size=8 is required — 96 must be divisible by patch_size.
#       The default patch_size=14 does NOT divide 96 (96/14=6.857 → crash).
#       patch_size=8  → 96/8=12 patches  ✓
#       patch_size=16 → 96/16=6 patches  ✓  (coarser, fewer compute)
python train.py data=billiards \
    img_size=96 \
    patch_size=8 \
    trainer.max_epochs=100 \
    wandb.enabled=False
```

Key config differences vs the default PushT run:

| Parameter | PushT | Billiards |
|-----------|-------|----------|
| `data` | `pusht` | `billiards` |
| `img_size` | 224 | 96 |
| `patch_size` | 14 | **8** (96 must be divisible by patch_size) |
| `data.dataset.frameskip` | 5 | 1 |
| `data.dataset.name` | `pusht_expert_train` | `billiards_expert_train` |
| Keys loaded | pixels, action, proprio, state | pixels, action, state |

---

## 4. Evaluate the Trained Model

```bash
cd ~/your-projects-folder/le-wm
source .venv/bin/activate

python eval.py \
    --config-name billiards \
    policy=<ckpt_name>
```

The eval config lives at `config/eval/billiards.yaml`. Key settings:

- `eval.img_size: 96` — matches the 96×96 billiards frames
- `eval.dataset_name: billiards_expert_train`
- `eval.eval_budget: 50` — steps per evaluation episode
- `output.filename: billiards_results.txt`

---

## 5. MPS Verification Script

```python
import torch
import torch.nn as nn

print(f"PyTorch version : {torch.__version__}")
print(f"MPS available   : {torch.backends.mps.is_available()}")
print(f"MPS built       : {torch.backends.mps.is_built()}")

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Active device   : {device}")

# Quick forward pass smoke-test
x = torch.randn(4, 3, 96, 96, device=device)
conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
out = conv(x)
print(f"Conv output     : {out.shape} on {out.device}  ✓")
```

Save as `verify_mps.py` and run with `python verify_mps.py`.
