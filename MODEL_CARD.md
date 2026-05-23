---
license: mit
base_model: lucas-maes/le-wm
tags:
- world-model
- jepa
- billiards
- pytorch
- mps
- apple-silicon
language:
- en
---

# LeWM Billiards — Trained World Model

> **Author:** Santosh Jaiswal ([@hellojais](https://huggingface.co/hellojais))  
> **Base architecture:** [LeWM](https://github.com/lucas-maes/le-wm) by Lucas Maes et al. (2025)  
> **Training data:** [hellojais/billiards-worldmodel](https://huggingface.co/datasets/hellojais/billiards-worldmodel)  
> **Code:** [hellojais/le-wm](https://github.com/hellojais/le-wm)  

## Model variants

| File | embed_dim | λ (SIGReg) | Best epoch | val/pred_loss | Notes |
|---|---|---|---|---|---|
| `lewm_epoch_8_object.ckpt` | 192 | 0.09 | 8 | 0.00946 | Best overall validation loss |
| `lewm_small_epoch_8_object.ckpt` | 32 | 0.01 | 8 | 0.00280 | Best prediction accuracy (2.6× better) |

## What this model learned

Trained on 4,000 episodes (971,321 frames) of 2D billiards gameplay.
The model learned to predict future frame embeddings from current 
embeddings and actions — encoding billiards physics purely from pixels.

**Probe results (lewm_small):**
- Target ball position: R²=0.988 ✅
- Cue ball position: R²=0.854 ✅  
- Ball velocities: R²=0.33–0.37 ⚠️

## Planning results

| Approach | Same-episode | Novel cross-episode |
|---|---|---|
| Pure JEPA embedding CEM | ❌ FAIL | ❌ FAIL |
| State-based hybrid CEM | ✅ SUCCESS (9 steps) | ✅ SUCCESS (13 steps) |

Pure JEPA planning failed due to uniform embedding geometry 
in this visually simple domain. See [FINDINGS.md](https://github.com/hellojais/le-wm/blob/main/FINDINGS.md) 
for complete analysis.

## Usage

```python
# Load checkpoint
import stable_worldmodel as swm
import torch

device = torch.device("mps")  # or "cuda" or "cpu"

# Load the small model (recommended)
checkpoint = torch.load(
    "lewm_small_epoch_8_object.ckpt", 
    map_location=device
)
```

## Training setup

- **Hardware:** Apple M5 Max (64GB unified memory)
- **Backend:** PyTorch MPS
- **Training time:** ~10 hours (10 epochs)
- **Framework:** PyTorch Lightning + stable-worldmodel

## Credits

Original LeWM architecture by:
Lucas Maes, Quentin Leroux, Gauthier Gidel, Glen Berseth  
Mila / McGill University (2025)  
[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
