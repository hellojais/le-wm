"""module_mamba.py — Mamba (Selective SSM) predictor for LeWM JEPA.

Drop-in replacement for ``ARPredictor`` in ``module.py``.  Identical call
interface; only the *internal architecture* changes.

Architecture difference vs. Transformer (ARPredictor)
──────────────────────────────────────────────────────
Transformer:
  • O(T²) causal self-attention over the T-frame context window (T=3 here).
  • Velocity must be inferred by attending to position *differences* between
    tokens.  With T=3 the model sees at most 2 adjacent position deltas
    simultaneously — barely enough for first-order velocity estimation.

Mamba (this file):
  • Maintains a PERSISTENT hidden state h ∈ ℝ^{d_inner × d_state} that is
    updated recurrently at each time step:

        h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t        (state accumulation)
        y_t = C_t · h_t  +  D ⊙ x_t              (output read-out)

    where  Ā_t = exp(Δ_t ⊗ A)  and  B̄_t = Δ_t ⊗ B_t  are obtained by
    *discretising* continuous-time matrices A and B using an input-dependent
    step size Δ_t  (the "selective" mechanism — Δ, B, C are all functions
    of the input token).

  • Velocity is encoded as the *change accumulated in h* across successive
    position embeddings, without needing an explicit attention window.  Even
    for T=3, h_3 contains a compressed history of how positions evolved over
    all three steps, giving a natural velocity signal.

  • The selective gate (Δ_t input-dependent) lets the model decide at each
    step how much of the previous state to retain vs. reset — useful for
    handling ball collisions (abrupt velocity changes).

Hypothesis: velocity R² (≈0.33 with Transformer) should rise because the
hidden state h explicitly integrates position differences into a persistent
velocity representation.

Implementation
──────────────
Pure PyTorch — no CUDA extensions.  Works on MPS (Apple Silicon), CPU, and
CUDA.  ``mamba-ssm`` is CUDA-only and therefore unsuitable for MPS; this
file implements the S6 core directly.

References
──────────
  Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces"  https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Selective State Space Model (S6 core)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Mamba S6 core — pure PyTorch, MPS/CPU/CUDA compatible.

    State equations (per time step t within a sequence):

        h_t  = Ā_t ⊙ h_{t-1}  +  B̄_t ⊙ x_t     ← persistent state update
        y_t  = C_t  ·  h_t    +  D   ⊙ x_t       ← output read-out

    Discretisation (ZOH, converts continuous A/B to discrete Ā/B̄):

        Ā_t [i,n] = exp( Δ_t[i] · A[i,n] )        shape (d_inner, d_state)
        B̄_t [i,n] = Δ_t[i] · B_t[n]               shape (d_inner, d_state)

    All of  Δ_t, B_t, C_t  are *input-dependent* (selective): they are linear
    projections of the current input token ``x_t``.  This allows the model to
    selectively decide what information to write into / read from the state.

    The sequential scan over T steps is O(T · d_inner · d_state); for the
    T=3 history window used here this is negligible compared to the MLP cost.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.d_state = d_state

        # dt_rank: controls expressiveness of the Δ projection
        dt_rank = max(1, d_model // 16)
        self.dt_rank = dt_rank

        # Pre-normalisation (applied before the projection, residual is added
        # *after* out_proj so the block is a proper pre-norm residual layer)
        self.norm = nn.LayerNorm(d_model)

        # ── Input projection ─────────────────────────────────────────────
        # Splits into x_ (processed by SSM) and z (multiplicative gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # ── Causal depthwise 1-D convolution ─────────────────────────────
        # Provides local temporal mixing *before* the SSM scan.
        # padding = d_conv-1, then trim → strictly causal output.
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            groups=d_inner,       # depthwise
            padding=d_conv - 1,
            bias=True,
        )

        # ── Input-dependent SSM parameter projections ────────────────────
        # x_proj maps each activated token to (dt_raw, B_ssm, C) jointly.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # dt_proj expands dt_raw from dt_rank → d_inner (one Δ per channel)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # ── Learned SSM matrices ─────────────────────────────────────────
        # A: diagonal state-decay matrix.
        #    Stored as log|A| (positive) so A_log > 0 always holds;
        #    actual A = -exp(A_log) is guaranteed negative-definite → stable.
        #    Initialised as log(1, 2, …, d_state) broadcast over d_inner.
        A_init = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_inner, -1)
            .contiguous()
        )
        self.A_log = nn.Parameter(torch.log(A_init))   # (d_inner, d_state)

        # D: skip / residual coupling (one scalar per inner channel)
        self.D = nn.Parameter(torch.ones(d_inner))     # (d_inner,)

        # ── Output projection ────────────────────────────────────────────
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)  — SSM output with residual added
        """
        residual = x
        B, T, _ = x.shape

        x = self.norm(x)

        # ── Project to inner dimension + gate ────────────────────────────
        xz = self.in_proj(x)                        # (B, T, 2·d_inner)
        x_, z = xz.chunk(2, dim=-1)                 # (B, T, d_inner) each

        # ── Causal 1-D convolution ───────────────────────────────────────
        # Conv1d expects (B, C, L); trim extra causal padding on the right.
        x_conv = x_.transpose(1, 2)                 # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[..., :T]       # causal trim
        x_ = F.silu(x_conv.transpose(1, 2))         # (B, T, d_inner)

        # ── Input-dependent SSM projections ─────────────────────────────
        proj_out = self.x_proj(x_)                  # (B, T, dt_rank + 2·d_state)
        dt_raw, B_ssm, C = proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ_t: (B, T, d_inner) — positive step size via softplus
        dt = F.softplus(self.dt_proj(dt_raw))
        # A: (d_inner, d_state) — negative definite
        A = -torch.exp(self.A_log.float())

        # ── Discretise: compute Ā and B̄ for every (batch, time) ────────
        # dA[b, t, i, n] = exp( dt[b,t,i] · A[i,n] )
        dA = torch.exp(
            dt.unsqueeze(-1)                        # (B, T, d_inner,      1)
            * A.unsqueeze(0).unsqueeze(0)           # (1,  1, d_inner, d_state)
        )                                           # (B, T, d_inner, d_state)

        # dB[b, t, i, n] = dt[b,t,i] · B_ssm[b,t,n]
        dB = (
            dt.unsqueeze(-1)                        # (B, T, d_inner,      1)
            * B_ssm.unsqueeze(2)                    # (B, T,       1, d_state)
        )                                           # (B, T, d_inner, d_state)

        # ── Sequential scan over T ───────────────────────────────────────
        # h is the PERSISTENT hidden state.  It accumulates position history
        # across steps so that velocity shows up as the *change in h*, rather
        # than needing explicit position differencing as in a Transformer.
        #
        # For T=3 (billiards history_size) this loop is just 3 iterations;
        # overhead is negligible and the implementation is MPS-portable.
        h = torch.zeros(
            B, self.d_inner, self.d_state,
            device=x.device, dtype=x_.dtype,
        )
        ys: list[torch.Tensor] = []
        for t in range(T):
            # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t  (state accumulation)
            h = dA[:, t] * h + dB[:, t] * x_[:, t].unsqueeze(-1)
            # y_t = C_t · h_t  (dot product over d_state dimension)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # (B, T, d_inner)

        # ── D-skip and SiLU gate ─────────────────────────────────────────
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_   # skip connection
        y = y * F.silu(z)                               # multiplicative gate

        # ── Output projection + residual ─────────────────────────────────
        return self.out_proj(y) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Mamba block: SSM + feedforward MLP
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """One full Mamba layer: SelectiveSSM followed by a feedforward MLP.

    The SSM handles temporal dynamics (persistent velocity encoding).
    The FFN adds per-position non-linear capacity (same role as in a
    Transformer block but without cross-token mixing).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_expand: int = 4,
    ):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_expand * d_model),
            nn.GELU(),
            nn.Linear(mlp_expand * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        x = self.ssm(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MambaPredictor — drop-in for ARPredictor
# ─────────────────────────────────────────────────────────────────────────────

class MambaPredictor(nn.Module):
    """Autoregressive next-state predictor using Mamba (Selective SSM).

    Drop-in replacement for ``ARPredictor`` — identical forward signature::

        forward(x, c)  →  (B, T, output_dim)
          x : (B, T, input_dim)   — projected encoder embeddings
          c : (B, T, input_dim)   — action conditioning embeddings
              (optional; defaults to zeros — for smoke-testing only)

    Constructor is intentionally positional-friendly so that the smoke test::

        MambaPredictor(num_frames=3, input_dim=32, hidden_dim=192, output_dim=32)

    and the train.py keyword-expansion call both work.

    Extra Transformer kwargs (``heads``, ``mlp_dim``, ``dim_head``) are
    silently absorbed via ``**kwargs`` so that configs written for
    ``ARPredictor`` can be passed without modification.

    Design
    ──────
    State embeddings (x) and action embeddings (c) are projected to
    ``hidden_dim`` and *added* before the Mamba stack.  This is equivalent
    to the AdaLN-zero conditioning in ARPredictor but simpler: the action
    modulates the *input* to the SSM scan rather than each layer's norm.

    The SSM scan then propagates both state history and action context through
    its persistent hidden state, learning to track velocity implicitly.
    """

    def __init__(
        self,
        num_frames: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        *,                          # everything below is keyword-only
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,       # accepted for API compatibility; unused
        emb_dropout: float = 0.0,
        **kwargs,                   # absorb ARPredictor-only kwargs: heads,
                                    # mlp_dim, dim_head — silently ignored
    ):
        super().__init__()
        output_dim = output_dim or input_dim

        # Learnable positional embedding — same as ARPredictor so both models
        # receive the same positional signal
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Project state embedding to hidden_dim
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim else nn.Identity()
        )

        # Project action embedding to hidden_dim (fused by addition at input)
        # Fusing at the input level gives the SSM full action context at every
        # scan step — analogous to AdaLN-zero in the Transformer version.
        self.act_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Mamba blocks operating at hidden_dim
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim else nn.Identity()
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)   — state embeddings from projector
            c: (B, T, input_dim)   — action embeddings (optional; zeros if None)
        Returns:
            (B, T, output_dim)
        """
        T = x.size(1)

        # Positional embedding — identical to ARPredictor
        x = x + self.pos_embedding[:, :T]
        x = self.emb_dropout(x)

        # Project to hidden_dim and fuse action conditioning
        x = self.input_proj(x)
        if c is not None:
            x = x + self.act_proj(c)
        # (if c is None — smoke-test path — we skip action conditioning)

        # Mamba blocks: each block's SelectiveSSM accumulates position
        # history into its persistent hidden state h, encoding velocity
        # as the change in h across time steps
        for block in self.blocks:
            x = block(x)

        return self.output_proj(self.norm(x))
