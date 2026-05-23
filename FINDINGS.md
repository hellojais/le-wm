# LeWM Billiards Planning — Research Findings

## Overview

We trained a JEPA-style world model on billiards expert demonstrations and
evaluated whether learned embeddings can serve as a planning signal via
Cross-Entropy Method (CEM) Model Predictive Control (MPC).

Four approaches were tested across three planning paradigms; only the state-based planner succeeded.

The probe experiment provides the sharpest diagnostic: the embedding *does* contain
accurate ball position (R²=0.988), but the dynamics predictor cannot simulate the
collision mechanics needed to pot a ball — so no cost function can rescue CEM.

---

## Experiments

### Evaluation Setup

**Task:** Pot the target ball into any pocket.

**Two evaluation combos:**
- **Combo A** — Same-episode: ep 500 frame 0 → ep 500 last frame (goal)
- **Combo B** — Cross-episode: ep 100 frame 0 → ep 3000 last frame (goal)

**Dataset:** `billiards_expert_train.h5` — 4000 episodes, 971,321 frames,
96×96×3 pixels, action range ≈ [−30.4, +30.0], mean ≈ 0, std ≈ 1.

---

### Experiment 1 — Embedding-based CEM, Original Model

**Model:** ViT-tiny encoder, `embed_dim=192`, `sigreg_weight=0.09`,
trained 20 epochs (best checkpoint: `lewm_epoch_19_object.ckpt`,
`val/pred_loss = 0.00737`).

**Planner:** CEM MPC, `PLAN_HORIZON=10`, `NUM_SAMPLES=1000`,
`N_CEM_ITERS=15`, `TOPK=100`, `CEM_CLAMP=30.0`.

**Cost function:** MSE between predicted final embedding and goal embedding.

**Results:**

| Combo | emb_dist_to_goal (range) | Final cost | Outcome |
|-------|--------------------------|------------|---------|
| A     | 18–22 (flat)             | 290        | FAILURE |
| B     | 18–22 (flat)             | 203        | FAILURE |

**Key diagnostic metrics:**
- Pairwise embedding L2: mean = 19.38 ± 1.56, CV = 0.081
- Min pairwise L2 = 10.16 (no clustering of similar frames)
- Prediction error (pred→real L2): 0.946

The embedding distance to goal never decreased across 150 steps — CEM received
no gradient signal.

---

### Experiment 2 — Embedding-based CEM, Small Model (Architectural Fix)

**Model:** Same ViT-tiny architecture, `embed_dim=32`, `sigreg_weight=0.01`,
trained 10 epochs (best checkpoint: `lewm_small_epoch_8_object.ckpt`,
`val/pred_loss = 0.00284`).

**Planner:** CEM MPC, `PLAN_HORIZON=10`, `NUM_SAMPLES=1000`,
`N_CEM_ITERS=20`, `TOPK=50`, `CEM_CLAMP=3.0` (corrected for std≈1 actions).

**Cost function:** Same — MSE between predicted final embedding and goal embedding.

**Results:**

| Combo | emb_dist_to_goal (range) | Final cost | Outcome |
|-------|--------------------------|------------|---------|
| A     | 8.2–8.3 (flat)           | ~70        | FAILURE |
| B     | 9.3–9.7 (flat → locked at 9.737) | ~94 | FAILURE |

**Key diagnostic metrics (vs original model):**
- Pairwise embedding L2: CV = 0.121 (vs 0.081) — less uniform
- Min pairwise L2 = 0.89 (vs 10.16) — similar frames now cluster
- Prediction error (pred→real L2): 0.111 (vs 0.946) — **8.5× better**

The architectural fix (smaller embed_dim + weaker sigreg regularisation)
significantly improved prediction accuracy and embedding structure. However,
the CEM cost still never approached zero, and `emb_dist_to_goal` remained flat
throughout. For Combo B, the ball reached a stopped configuration from step 84
onward and the planner locked into a fixed cost of ~93.9.

---

### Experiment 3 — State-based CEM (No World Model)

**Planner:** CEM runs rollouts directly in the billiards simulator (no neural
network involved in cost computation). Cost = minimum distance from target ball
to any pocket across the simulated trajectory.

**Parameters:** `PLAN_HORIZON=20`, `EXEC_PER_PLAN=5`, `MAX_EVAL_STEPS=300`,
`NUM_SAMPLES=500`, `N_CEM_ITERS=20`, `TOPK=50`, `ACT_LO=-30.0`, `ACT_HI=30.0`.

**Results:**

| Combo | Steps to success | Min dist to pocket | Outcome    |
|-------|------------------|--------------------|------------|
| A     | 9                | 0.02 px            | **SUCCESS** |
| B     | 13               | 0.39 px            | **SUCCESS** |

This approach works because the cost function is directly informative about the
task (ball position relative to pockets), and the simulator is fast enough for
CEM rollouts without requiring a learned model.

---

### Experiment 4 — Probe-based CEM (Linear Probe on Frozen Embeddings)

**Probe architecture:** 2-layer MLP, `embed_dim(32) → hidden(64) → state(10)`.

**Training:** 1000 random frames from the dataset, 500 epochs, Adam lr=1e-3.

**Probe training results:**

| Epoch | MSE    | Mean R² |
|-------|--------|---------|
| 0     | 1.027  | −0.003  |
| 100   | 0.392  | 0.611   |
| 200   | 0.362  | 0.641   |
| 300   | 0.340  | 0.661   |
| 400   | 0.327  | 0.676   |
| 499   | 0.314  | 0.687   |

**Per-dimension R² (final probe on training data):**

| Dimension | R²     | Quality   |
|-----------|--------|-----------|
| cue_x     | +0.854 | good      |
| cue_y     | +0.731 | moderate  |
| cue_vx    | +0.351 | poor      |
| cue_vy    | +0.325 | poor      |
| tgt_x     | +0.988 | excellent |
| tgt_y     | +0.976 | excellent |
| tgt_vx    | +0.366 | poor      |
| tgt_vy    | +0.359 | poor      |
| pkt_x     | +0.981 | excellent |
| pkt_y     | +0.937 | excellent |
| **mean**  | **0.687** |        |

Target ball position (the planning-relevant signal) is decoded with R²=0.99/0.98.
The embedding *does* contain accurate ball position information.

**CEM cost function:** `||probe(predicted_emb_last) − goal_state_norm||²`

**Planning results:**

| Combo | probe_dist (range) | Best cost | Outcome |
|-------|-------------------|-----------|---------|
| A     | 2.73–2.79 (flat)  | 7.354     | FAILURE |
| B     | 1.85–1.87 (flat)  | 1.573     | FAILURE |

`probe_dist` was completely flat in both combos — no action sequence moved the
decoded predicted state closer to the goal.

---

### Velocity Encoding Insight

The per-dimension R² breakdown reveals a systematic split between position and velocity:

| Signal type | Dimensions      | R² range     | Interpretation |
|-------------|-----------------|--------------|----------------|
| Position    | tgt_x, tgt_y    | 0.976–0.988  | Excellent — model knows where the ball is |
| Position    | pkt_x, pkt_y    | 0.937–0.981  | Excellent — model knows pocket locations |
| Position    | cue_x, cue_y    | 0.731–0.854  | Good — cue position well encoded |
| Velocity    | cue_vx, cue_vy  | 0.325–0.351  | Poor — cue velocity barely encoded |
| Velocity    | tgt_vx, tgt_vy  | 0.359–0.366  | Poor — target velocity barely encoded |

**Why this matters for billiards specifically:**

Potting a ball is a momentum-transfer problem. Success depends critically on:
1. The *velocity* of the cue at impact (determines transfer force)
2. The *direction* of the resulting target velocity (determines pocket angle)
3. The *speed* after collision (determines whether the ball reaches the pocket)

A world model that barely encodes velocity (R²≈0.35) cannot reliably predict
whether a given cue shot will pot the target ball. Even if positions are known
precisely, a collision event produces a discontinuous velocity change that the
model must predict accurately to be useful for planning. The model fails at this.

**Why single-frame embeddings can't encode velocity:**

Velocity is inherently a temporal quantity — it cannot be read from a single
frame without motion blur or explicit optical flow. JEPA encodes individual frames;
the only velocity information available is implicit in the *pattern* of positions
across the history context (3 frames), which is insufficient for precise momentum.

---

## Comparison Table

| Approach               | Model used       | Cost function               | Combo A     | Combo B     | Steps   |
|------------------------|------------------|-----------------------------|-------------|-------------|---------|
| Embedding CEM (orig)   | embed_dim=192    | MSE in embedding space      | FAILURE     | FAILURE     | 150     |
| Embedding CEM (small)  | embed_dim=32     | MSE in embedding space      | FAILURE     | FAILURE     | 150     |
| State-based CEM        | None (simulator) | Dist to pocket              | **SUCCESS** | **SUCCESS** | 9 / 13  |
| Probe-based CEM        | embed_dim=32     | MLP(emb)→state dist²        | FAILURE     | FAILURE     | 150     |

---

## Root Cause Analysis

### Why embedding-based CEM fails

**The fundamental problem is not prediction quality — it is cost landscape structure.**

JEPA trains the predictor to minimise MSE between predicted and actual future
embeddings. This objective does not require that:

1. L2 distance in embedding space ∝ semantic distance between scenes, or
2. Actions that move the ball toward a pocket also move predicted embeddings
   toward the goal embedding.

In other words, the embedding space may accurately predict *what the next frame
looks like* without organising itself so that CEM can navigate from start to
goal by following a cost gradient.

**Evidence:**
- The small model's `emb_dist_to_goal` stayed flat at ~8–10 throughout all 150
  steps, even as the CEM searched over 1000 × 20 = 20,000 action sequences per
  step. The cost landscape is flat — no action sequence finds a path that
  decreases embedding distance to the goal.
- The original model's embedding space was more uniform (CV=0.081, min
  pairwise L2=10.16): literally no clustering of semantically similar frames.
- The small model improved this (CV=0.121, min pairwise L2=0.89) but not
  enough to provide a navigable gradient for CEM.

### Why the architectural fix helped prediction but not planning

Reducing `embed_dim` (192→32) forces the encoder to discard redundant
dimensions and represent only the most informative features. Reducing
`sigreg_weight` (0.09→0.01) weakens the regularisation that pushes all
embeddings toward a fixed Gaussian — allowing the embedding space to develop
more structure.

These changes improved:
- Prediction accuracy 8.5×
- Minimum pairwise L2 from 10.16 to 0.89 (similar frames cluster)

But JEPA's training signal is "predict the next frame correctly," not "organise
embeddings so that L2 distance reflects task-relevant state similarity." The
necessary alignment between embedding geometry and task geometry was not
learned.

### Why the probe also failed (key finding)

The linear probe experiment provides the critical diagnosis.

**What the probe confirmed:**
- The embedding *does* contain ball position information: R²=0.988 for `tgt_x`,
  R²=0.976 for `tgt_y`, R²=0.981 for nearest pocket x.
- Decoding state from a real frame's embedding works well.

**What the probe revealed:**
- `probe_dist` stayed flat throughout all 150 steps for both combos — identical
  to the raw embedding failure.
- The best CEM plan in Combo B achieved cost=1.57 (sqrt≈1.25 normalised units
  from goal state) and never approached 0.

The bottleneck is not the cost function — it is the **dynamics model's predicted
embeddings**. When the predictor produces `predicted_emb` for a 10-step action
sequence, those embeddings do not correspond to physically accurate future states
of the billiards environment. Specifically:

- The predictor cannot simulate the contact mechanics needed to pot a ball.
  Potting requires a precise cue → target → pocket alignment; the predictor
  likely learned average-case dynamics rather than the sharp nonlinearity of
  collision.
- The minimum achievable CEM cost (~1.57) is a floor imposed by the model —
  no action sequence in the search space causes the predicted decoded state
  to reach a potted configuration.
- This is a **model exploitation / Dyna failure mode**: CEM finds sequences
  that look good under the model but don't correspond to physically achievable
  outcomes.

**Conclusion:** The world model is insufficiently accurate at predicting the
consequence of actions over a 10-step horizon for CEM to exploit it. The
state-based planner succeeds because it runs the *real* dynamics — no model
error. Any JEPA-based planner will fail until the model can accurately predict
the result of a potting shot.

---

## What Would Be Needed to Fix Pure JEPA Planning

### Option 1 — Linear probe on top of frozen embeddings ✗ TRIED, FAILED

Trained a 2-layer MLP `f: embedding → state_vector` on 1000 labelled frames.
Used `||f(predicted_emb) - goal_state||` as the CEM cost.

- Result: FAILURE. `probe_dist` flat at ~1.85–2.78 for all 150 steps.
- The embedding *does* contain state information (R²=0.988 for tgt_x), but the
  *predicted* embeddings from the dynamics model don't correspond to physically
  accurate future states. The model cannot imagine a trajectory that pots the ball.
- Root cause: dynamics model accuracy, not cost function design.

### Option 2 — Frame stacking for implicit velocity encoding

The probe experiment showed velocity R²≈0.35 — the model barely knows ball speed
or direction. The predictor receives 3 history frames, but the ViT encoder processes
each independently. Stacking multiple frames as channels before encoding would give
the encoder direct access to optical flow:

```python
# Instead of: encoder(frame_t) → emb_t
# Do:         encoder(cat([frame_{t-2}, frame_{t-1}, frame_t])) → emb_t
```

This is cheap (no architecture change to the predictor) and should raise velocity
R² significantly, which is necessary (though not sufficient) for planning to work.

### Option 3 — Auxiliary velocity prediction head

Add a lightweight head on top of the encoder that explicitly predicts
`[vx, vy]` for both balls and train it jointly with the JEPA loss:

```
L_total = L_jepa + λ_vel · MSE(vel_head(emb), gt_velocity)
```

This forces the embedding to represent velocity explicitly, which the probe showed
it currently lacks. The head can be discarded at inference — its only role is to
shape the embedding during training.

### Option 4 — Contrastive or VICReg objective

Replace or augment the JEPA prediction loss with a loss that explicitly
aligns embedding distance with state distance:

```
L_total = L_jepa + λ · ||d_emb(s1,s2) - d_state(s1,s2)||²
```

This directly trains the embedding space to be metrically aligned with the
task-relevant state, fixing the flat cost landscape.

### Option 5 — Contrastive loss over goal-reachable pairs

A targeted variant: sample pairs of states where one can reach the other
(from expert trajectories, they are adjacent in time or within the same episode)
and pairs that cannot (different episodes). Train:

```
L_contrastive = push apart(cross-episode pairs) + pull together(same-episode pairs)
```

This would specifically align the embedding space with the notion of
"reachability" rather than raw state similarity, which is exactly what CEM needs.

### Option 6 — Goal-conditioned JEPA

Condition the predictor on a goal embedding and train with a reach reward.
This requires adding a goal-conditioning head and modifying the training
pipeline, but produces an embedding space explicitly structured for planning.

### Option 7 — Hybrid cost function (pure accuracy fix)

Use the world model for dynamics rollouts (predicting future frames cheaply)
but replace the CEM cost with the real simulator check on the decoded state:

```
cost = state_decoder(predicted_emb[-1]) → ball_position → dist_to_pocket
```

This would work only if the dynamics predictor becomes accurate enough to
imagine a potting shot. The current model cannot — this requires either more
data, more capacity, or better training objectives (Options 2–5).

---

## Conclusion — When Does JEPA Planning Work vs Fail?

### When it works

JEPA-based CEM planning is likely to succeed when:

1. **The task has slow, smooth dynamics** — where position-only prediction
   is sufficient and velocity doesn't dominate outcomes (e.g., navigation,
   slow manipulation).
2. **The embedding space is metrically aligned with the goal** — either by
   design (contrastive objectives) or because the task is simple enough that
   prediction accuracy alone induces alignment.
3. **The planning horizon is short relative to the dynamics timescale** — a
   1–2 step prediction is much more accurate than a 10-step one for any model.
4. **The goal is reachable from many action sequences** — CEM does not require
   a precise gradient, only a cost landscape that is "roughly correct" so that
   elite samples cluster near the right direction.

### Why it fails here

Billiards is a maximally hostile test case for JEPA planning:

| Property | Why it's hard for JEPA |
|---|---|
| Collision dynamics | Sharp nonlinearity — one frame before/after contact looks nearly identical; the model must predict the exact post-collision velocity |
| Potting is rare | Only ~1 frame out of a typical 100-step episode is the potting frame; the model rarely learns this regime |
| Velocity is invisible | Single frames don't encode velocity; the model can't reason about momentum |
| Goal is a specific microstate | Reaching a pocket requires a precise shot; many trajectories are "close" in pixel space but physically fail |
| Long effective horizon | The cue must interact with the target which must interact with the pocket — a 3-step causal chain within 10 model steps |

### What the four experiments establish

1. **Embedding geometry alone cannot drive planning** (Experiments 1–2): Even
   with 8.5× better prediction accuracy and explicit embedding structure fixes,
   raw L2 distance in embedding space provides no useful gradient for CEM.

2. **A good cost function cannot rescue a bad dynamics model** (Experiment 4):
   The probe achieved R²=0.988 for target position — near-perfect state
   decoding — yet planning still failed completely. The bottleneck was the
   predictor, not the cost.

3. **The real simulator solves it trivially** (Experiment 3): State-based CEM
   with the true dynamics pots the ball in 9–13 steps. The task is plannable;
   only the learned model is the obstacle.

**The fundamental lesson:** JEPA learns to predict *plausible* futures, not
*physically accurate* futures. For tasks where approximate prediction is
enough (e.g., "is there an obstacle ahead?"), JEPA planning works. For tasks
where the outcome is determined by a precise physical event (a collision), the
approximate model is worthless for planning regardless of cost function design.

---

---

## Extended Experiments: Isolating the Root Cause

### Experiment 5 — Mamba Predictor (Stateful Architecture)

**Hypothesis:** The velocity encoding gap is architectural — the Transformer
predictor's fixed 3-frame window can't accumulate velocity history. A Mamba
predictor with persistent hidden state should fix it.

**Setup:** Replaced ARPredictor (Transformer) with a pure PyTorch S6 Mamba
implementation (MPS-compatible). All other settings identical to
`billiards_small` config (`embed_dim=32`, `lr=5e-5`, 10 epochs).

**Results (three-way comparison, consistent 300ep/1000-frame probe):**

| Metric | Transformer | Mamba | Change |
|---|---|---|---|
| val/pred_loss | 0.0035 | 0.0034 | −3.1% |
| velocity R² (mean, 4 dims) | 0.296 | 0.297 | +0.3% |
| position R² (tgt mean) | 0.983 | 0.983 | ≈0% |

**Conclusion:** Architecture is NOT the bottleneck. Velocity R² improved by
only 0.3% — statistically negligible and within probe noise. The bottleneck
is upstream of the predictor, in the encoder or the objective itself.

---

### Experiment 6 — Frame Stacking (9-channel input)

**Hypothesis:** The ViT encoder processes frames independently, so velocity
is only implicit in the 3-frame history context. Stacking frames [t−2, t−1, t]
as 9 input channels gives the encoder explicit optical flow — velocity should
be directly visible in the first layer.

**Setup:** Stacked frames [t−2, t−1, t] as 9 channels. ViT encoder
re-initialised with `num_channels=9` (no pretrained weights). All other
settings identical to `billiards_small` config. Best checkpoint: epoch 7
(`val/pred_loss=0.005935`).

**Results (consistent short probe — 300 epochs, 1000 frames — same as Exp 5):**

| Metric | Transformer | Frame Stack | Change |
|---|---|---|---|
| val/pred_loss | 0.0035 | 0.0087 | +149% worse |
| velocity R² (mean, 4 dims) | 0.296 | 0.286 | −3.4% |
| target velocity R² (tgt_vx + tgt_vy mean) | 0.319 | 0.399 | +0.080 |
| cue velocity R² (cue_vx + cue_vy mean) | 0.273 | 0.174 | −0.099 |
| position R² (tgt mean) | 0.983 | 0.579 | −0.404 |

**Extended probe diagnostic — FrameStack only (1000 epochs, 5000 frames):**

To distinguish "probe didn't converge" from "information is not in the
embedding," the probe was re-run with 5× more data and 3× more epochs on the
FrameStack model only. Position R² plateaued at 0.590 — only +0.011 above the
short-probe result of 0.579 — confirming the information is genuinely absent.

The extended probe also reveals full per-dimension specialisation:

| Dimension | Transformer (short probe) | Frame Stack (extended probe) | Δ |
|---|---|---|---|
| tgt_vx | 0.336 | 0.771 | +0.435 |
| tgt_vy | 0.301 | 0.753 | +0.452 |
| cue_vx | 0.297 | 0.048 | −0.249 |
| cue_vy | 0.249 | 0.074 | −0.175 |
| tgt_x | 0.988 | 0.484 | −0.504 |
| tgt_y | 0.978 | 0.697 | −0.281 |

*Note: FrameStack values above are from the 1000-epoch extended probe; the
short-probe equivalents for FrameStack are tgt_vx=0.372, tgt_vy=0.426,
cue_vx=0.186, cue_vy=0.161, showing the same directional pattern but at
lower magnitude due to probe underfitting.*

**Key finding: JEPA's prediction objective creates representational eviction
under motion-signal pressure.**

With explicit optical flow available via 9-channel input, the JEPA loss rewards
encoding target ball velocity over absolute position — because velocity is more
predictive of the next embedding. The model made a rational optimisation choice
that destroys goal-directed planning capability.

Specifically:
- Target ball velocity (struck by the expert) → maximised (R²≈0.76)
- Cue ball velocity (already supplied via `action_encoder`) → evicted (R²≈0.06)
- Absolute ball positions → partially evicted (R²≈0.58 vs 0.98 baseline)

The short probe confirms the directional effect is real; the extended probe
confirms it is not a probe artifact — the information is genuinely absent from
the 32-dimensional embedding.

---

## Unified Summary Across All Six Experiments

All velocity and position R² values below are from the three-way comparison
script (`compare_predictors.py`, 300-epoch / 1000-frame probe, seed=42).
Extended probe results for Exp 6 are noted separately.

| Experiment | Key change | vel R² (mean) | pos R² (tgt) | Finding |
|---|---|---|---|---|
| Exp 2 — Small model (baseline) | embed_dim=32 | 0.296 (300ep probe) | 0.983 | reference |
| Exp 5 — Mamba predictor | stateful architecture | 0.297 | 0.983 | not the bottleneck |
| Exp 6 — Frame stacking | 9-channel input | 0.286 (short probe) | 0.579 | eviction discovered |
| Exp 6 — Extended probe | better evaluation | 0.412 (1000ep/5000fr) | 0.590 | eviction confirmed |

**The root cause is a fundamental tension in JEPA's single prediction objective:**

JEPA optimises for next-embedding predictability. In visually simple environments
with explicit motion signal, velocity becomes MORE predictive than position. The
model rationally evicts position encoding in favour of velocity encoding — which
destroys the spatial structure needed for goal-directed planning.

This is not a failure of JEPA per se. It is a precise boundary condition:
JEPA's single objective cannot simultaneously optimise for:

1. **Next-state predictability** (training objective)
2. **Goal-relevant spatial completeness** (planning requirement)

In Push-T, these objectives align — position IS highly predictive of the next
position. In billiards with frame stacking, they diverge: the struck ball's
velocity overwhelms the planning-critical absolute position signal.

---

## Implementation Notes

- All experiments run on Apple Silicon M5 Max (MPS backend).
- Training: `precision="32"`, `pin_memory=False`, `num_workers=0` required for MPS.
- Dataset actions are already normalised (mean≈0, std≈1 per dimension).
  CEM clamp of ±3.0 covers ~99.7% of the action distribution.
- BilliardsEnv: `WINDOW_SIZE=512`, `RENDER_SIZE=96`, `FRICTION=0.96`,
  `MAX_STEPS=300`, `POCKET_RADIUS=20`, pockets at corners (~20,20), (~492,20),
  (~20,492), (~492,492).
- State vector: `[cue_x, cue_y, cue_vx, cue_vy, target_x, target_y,
  target_vx, target_vy, nearest_pocket_x, nearest_pocket_y]`.
