"""module_auxloss.py — Auxiliary state supervision head for JEPA training.

Adds a lightweight linear head on top of the ViT CLS token (192-dim) that
predicts the ground-truth state vector during training.  The head is discarded
at inference — its sole purpose is to prevent the encoder from evicting
position information under JEPA's prediction objective.

Usage in train.py:
    from module_auxloss import AuxStateHead
    aux_head = AuxStateHead(hidden_dim=192, state_dim=10)
    world_model.aux_head = aux_head          # attach to JEPA model

Loss (computed in lejepa_forward):
    aux_loss = MSE(aux_head(cls_token_flat), state_flat)
    total_loss = pred_loss + λ_sigreg * sigreg_loss + λ_aux * aux_loss
"""

from __future__ import annotations

import torch.nn as nn


class AuxStateHead(nn.Module):
    """Linear head: CLS token (hidden_dim) → state vector (state_dim).

    Deliberately kept minimal — one linear layer with LayerNorm on the input.
    A heavier MLP would risk overfitting on the 192-dim space and dominating
    the JEPA loss gradient.

    Parameters
    ----------
    hidden_dim:
        Dimension of the ViT CLS token (encoder.config.hidden_size, default 192).
    state_dim:
        Dimension of the state vector to predict (default 10 for billiards).
    """

    def __init__(self, hidden_dim: int = 192, state_dim: int = 10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        """
        x : (N, hidden_dim)  — flat batch of CLS tokens
        returns : (N, state_dim)
        """
        return self.linear(self.norm(x))
