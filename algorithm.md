## Simplified SAM2 TTA (Pseudo-label + LoRA + Entropy/Consistency)

### Inputs
- Test image `I`
- Base pipeline `BasePipeline(...)` → pseudo label mask `M_hat` and prompt `P`
- SAM2 predictor (student) `f_theta` with LoRA on decoder only
- Config: `T` steps, loss weights `lambda_e, lambda_c`, augment family `g(...)`

### Outputs
- Adapted LoRA params `theta_T` (per-image; reset for next image)
- Final prediction `S*`

---

### Step 0 — Pseudo-label supervision (once)
1) Run base pipeline on `I`: `M_hat, P <- BasePipeline(I)`  
2) Freeze prompts for TTA: use `P` as-is (no re-search).

### Step 1 — LoRA fine-tuning (T steps)
For `t = 1..T`:
1) Predict on original view: `S0 = f_theta(I, P)`  
2) Sample two augmented views `g_v` (scale/flip), predict `S_v`, warp back `S_v_hat = g_v^{-1}(S_v)`  
3) Losses:
   - Pseudo-label loss: `L_sup = BCE(S0, M_hat)`  
   - Entropy loss (weightable): `L_entropy = H(S0)`  
   - Consistency loss (weightable): `L_cons = 1 - SoftDice(S0, S_v_hat)`  
4) Total loss: `L = L_sup + lambda_e * L_entropy + lambda_c * L_cons`  
5) Backprop on LoRA params only; optimizer step.

### Step 2 — Final output
- `S* = f_theta(I, P)` after `T` steps; reset LoRA for next image.
