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
- frame-stacking
- auxiliary-supervision
language:
- en
---

# LeWM Billiards — Trained World Model

> **Author:** Santosh Jaiswal ([@hellojais](https://huggingface.co/hellojais))  
> **Base architecture:** [LeWM](https://github.com/lucas-maes/le-wm) by Lucas Maes et al. (2025)  
> **Training data:** [hellojais/billiards-worldmodel](https://huggingface.co/datasets/hellojais/billiards-worldmodel)  
> **Code:** [hellojais/le-wm](https://github.com/hellojais/le-wm)  

## Model variants

| File | embed_dim | Input | λ_aux | Best epoch | val/pred_loss | Notes |
|---|---|---|---|---|---|---|
| `lewm_epoch_8_object.ckpt` | 192 | 3-ch | — | 8 | 0.00946 | Original full-size transformer |
| `lewm_small_epoch_8_object.ckpt` | 32 | 3-ch | — | 8 | 0.00280 | Transformer baseline |
| `lewm_mamba_best_object.ckpt` | 32 | 3-ch | — | best | 0.00340 | Mamba predictor |
| `lewm_framestacked_best_object.ckpt` | 32 | 9-ch | — | 7 | 0.00594 | Frame-stacking; JEPA eviction occurs |
| `lewm_auxloss_full_best_object.ckpt` | 32 | 9-ch | 0.1 | 7 | **0.00105** | Aux state supervision; eviction fixed; best model |

## What this model learned

Trained on 4,000 episodes (971,321 frames) of 2D billiards gameplay.
The model learned to predict future frame embeddings from current 
embeddings and actions — encoding billiards physics purely from pixels.

**Probe results** (linear probe on encoder representations; 192-dim = pre-projector CLS token, 32-dim = post-projector):

| Model | Rep dim | pos R² | vel R² |
|---|---|---|---|
| `lewm_small` | 32 | 0.983 | 0.296 |
| `lewm_mamba` | 32 | 0.983 | 0.297 |
| `lewm_framestacked` | 192 | 0.446 ⚠️ | 0.138 |
| `lewm_framestacked` | 32 | 0.599 | 0.417 |
| `lewm_auxloss_full` | 192 | **0.999** ✅ | **0.947** ✅ |
| `lewm_auxloss_full` | 32 | 0.982 | 0.554 |

> **Key finding:** Frame-stacking causes JEPA representational eviction — the ViT encoder stops encoding ball position (pos R²=0.446 at 192-dim) under optical-flow pressure. Adding a lightweight auxiliary state supervision head (λ=0.1) fully recovers position encoding (pos R²=0.999) and achieves the best prediction loss across all variants.

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
- **Training time:** ~10–11 hours per run (10 epochs); 4 model variants trained
- **Framework:** PyTorch Lightning + stable-worldmodel

## Credits

Original LeWM architecture by:
Lucas Maes, Quentin Leroux, Gauthier Gidel, Glen Berseth  
Mila / McGill University (2025)  
[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
