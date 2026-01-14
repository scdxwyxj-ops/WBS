## SAM2 Pipeline Only (No TTA)

### Inputs
- Test image `I`
- Pipeline config `C` (SLIC, graph nodes, sampling rules, thresholds, selection strategy)
- SAM2 predictor `F`

### Outputs
- Final mask `M*`
- Optional mask pool `P = {M_i, score_i}`

---

### Textbook Flowchart (Pipeline)

```
┌──────────────┐
│  Input I, C  │
└──────┬───────┘
       │
       v
┌────────────────────────────┐
│ Preprocess (resize, SLIC)  │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Build graph over segments  │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Init prompts (center)      │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ SAM2 predict logits/mask   │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Score + add to mask pool   │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Update prompts (boundary)  │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Loop until max_iters       │
└──────┬─────────────────────┘
       │
       v
┌────────────────────────────┐
│ Select final mask M*       │
└────────────────────────────┘
```

---

### Algorithm Steps

1) **Preprocess**  
Resize image to target size, run SLIC, build segment graph.

2) **Initialize prompts**  
Pick initial positive points near the image center (configurable range).

3) **Iterative expansion**  
Repeat up to `max_iterations`:  
   - Run SAM2 → logits, mask, score  
   - Add mask + score to pool  
   - Update prompts using confident boundary points

4) **Mask pool selection**  
Choose final `M*` by config strategy:  
   - `score_top_k` (pick best score), or  
   - `cluster_middle` (select middle cluster, then best score)

5) **Output**  
Return final mask `M*` and optional pool for analysis.
