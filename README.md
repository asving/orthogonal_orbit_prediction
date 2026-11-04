# How a transformer learns orthogonal dynamics in context

**Task.** Each sequence has its own random orthogonal matrix U (6×6, Haar) and unit vector v. The model sees x₁ = Uv, x₂ = U²v, … and predicts the next vector (MSE). The Bayes-optimal predictor is least squares in token space:

    x̂_{t+1} = Σ_j c_j x_j ,   c_j = ⟨x_{j-1}, S⁺ x_t⟩ ,   S = Σ_{k<t} x_k x_kᵀ   (context covariance)

U is never formed. Writing x_t as a combination of earlier tokens and shifting each by one step gives U x_t.

**Mechanism found** (2-4 layer transformers, d=128, 4 heads, with and without MLPs):

1. A layer-0 **previous-token head** pairs each value x_j with the key x_{j-1}.
2. Layer-0 **similarity heads** (attention logits ∝ ±⟨x_t, x_j⟩) write S x into the residual stream at every position.
3. Layer-1 **regression heads** come as a sign-split pair (softmax weights are non-negative, regression coefficients are not). They attend with dot products ⟨x_t, x_{j-1}⟩ preconditioned by S x, a truncated Neumann series for S⁺. Their output maps are ±(multiple of the identity) on x_j.
4. **Later layers** take gradient-descent-like steps: their values carry x_j minus the model's earlier prediction of x_j, and their causal pair weights move the total weight toward c.
5. Nothing stores U, S, S⁺x or the characteristic polynomial of U (linear probes R² ≤ 0.18). The model's state is the context itself, read by attention.

**Causal tests** (`python causal.py <model>`, outputs in `results/`):

| test | what it does | result (t = 12 tokens unless noted) |
|---|---|---|
| 1. value-path patching | perturb x_j only where it is read as a value; the prediction's response is the causal pair weight w_j (least squares: w_j = c_j) | layer-1 weights are a blend, w ≈ 0.2–0.3 c + 0.15–0.35 s (s = raw dot product). In the 3- and 4-layer models, weights added by layers ≥ 2 correlate with the layer-1 shortfall c − w (0.75 at t=4, 0.42–0.57 at t=12) |
| 2. query patching | donor trajectory reflected so its current token equals ours, different context; patch non-previous-token layer-0 heads at the query | error vs least squares rises 0.014 → 0.27 (L2). The shift aligns with the first-order preconditioning prediction −M(S′−S)x_t (cos +0.30, 79% positive); the previous-token-head control does not (cos −0.10). Alignment ≈ 0 in the 4-layer attention-only model |
| 3. attention replacement | replace layer-1 attention by fitted idealised patterns | L2: real 0.013, pure dot product 0.098, S-preconditioned 0.044. L2 attention-only: 0.023 / 0.123 / 0.060. Preconditioning recovers about two thirds; deeper models' layer 1 is less cleanly bilinear |

**Not yet pinned down.** About a third of the layer-1 pattern is a per-query adaptive correction that no fixed bilinear form captures (candidate: softmax normalisation acting as a data-dependent step size). Value-path weights still differ from c with fewer than n tokens, which may partly reflect that the perturbation leaves the manifold of valid trajectories (it breaks inner-product invariants).

## RNNs on the same task (what we are confident of)

Single-layer LSTM and GRU (hidden 256) reach late MSE 0.012 / 0.015. A vanilla RNN reaches 0.106. Least squares reaches 0.

For any U, Cayley–Hamilton gives x_{t+1} = Σ_{k=0..5} a_k x_{t-k} with a = −(characteristic-polynomial coefficients of U). These coefficients depend on U only, not on v. For the LSTM and GRU:

1. **U, S and S⁺ are not stored.** Linear probes of the hidden state give R² ≤ 0.3 for U and ≈ 0 for S and S⁺.
2. **The prediction is this lag recurrence.** The effective coefficients a_t = W_t⁻¹ x̂_{t+1} (W_t = the last 6 tokens) converge to the characteristic-polynomial coefficients. Their distance from them falls from 1.8 at t=5 to 0.2 at t=23 (|a| ≈ 2).
3. **The coefficients do not depend on v.** Running the same U from 8 different starting vectors gives coefficients with spread 0.06 at t=22. The spread of the true coefficients across different U is 6.9.
4. **The coefficients are in the hidden state.** A linear probe reads the effective a_t with R² ≈ 0.9 (sequences with well-conditioned windows).

The transformer regresses over token positions with the context as its memory. The RNN learns the coefficients of the recurrence over lags, which depend on U only.

**Not pinned down:** the rule that updates the coefficients. On training-distribution sequences, the predictions match regularised least squares over lags using the last one or two windows. That test cannot tell candidate rules apart, because one window already determines a. After a splice to another sequence, the RNN recovers over more than 10 steps. Least squares with forgetting, normalised LMS and step-capped variants all recover in about 6.

**Files.** `common.py` task, model with intervention hooks, least-squares reference. `train.py` (`python train.py 2` or `python train.py 4 --attnonly`). `causal.py` the three tests. `models/` trained weights.
