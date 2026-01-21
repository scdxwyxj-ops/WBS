# Superpixel-Guided Iterative Prompting (SGIP)

> Default choice in the final selection: **pick the largest-area mask in the middle cluster**.

---

## 1. Inputs / Hyper-parameters

- Image: \(I\)
- Resize long side: \(L\) (default: 260). Short side scaled proportionally.
- SLIC:
  - number of superpixels: \(K\) (default: 36; tuned by greedy search)
  - compactness: \(c\)
  - color space: RGB
- Center window ratio: \(\alpha \in (0, 0.5)\) (default range you use: 0.3–0.7, i.e. \([\alpha,1-\alpha]^2\))
- Init prompt counts:
  - positives: \(n_+\)
  - negatives: \(m_-\)
- Max iterations: \(T\) (safety cap)
- Optional Top-\(q\) rerun: \(q\) (default: 3)
- Optional non-convex threshold: \(\tau_{\text{convex}}\) (default: 0.85)
- Optional positive-point prune distance: \(d_{\min}\)
- Mask sequence de-dup IoU threshold: \(\tau_{\text{iou}}\)
- Clustering: k-means clusters \(k=3\)

---

## 2. Notation

- Resized image: \(I' = \mathrm{ResizeLongSide}(I, L)\)
- Superpixel label map: \(S \in \{1,\dots,K\}^{H'\times W'}\)
- Superpixel pixel set: \(\Omega_j = \{x \mid S(x)=j\}\)
- Superpixel centroid (float): \(\mu_j\)
- Superpixel mean grayscale: \(g_j\)
- Prompt point set: \(P = \{(u_i, y_i)\}\), \(y_i \in \{+1,-1\}\)
- Mask prompt: \(M^{\text{prompt}} \in \{0,1\}^{H'\times W'}\)

**SAM2 inference (given prompts):**
- \((\ell, M, s) \leftarrow \mathrm{SAM2}(I', M^{\text{prompt}}, P)\)
  - \(\ell\): logits map
  - \(M\): output mask
  - \(s\): SAM2 output score (quality score for mask)

**Logits thresholding:**
- \(\hat{M} = \mathbb{1}[\ell > 0]\) (equivalently prob > 0.5)

**Probability map:**
- \(p(x) = \sigma(\ell(x))\)

**Soft vote (mean prob in superpixel):**
- \(v_j = \frac{1}{|\Omega_j|}\sum_{x\in\Omega_j} p(x)\)

---

## 3. Deterministic point placement (A, B fixes)

To place a prompt point for a region \(R\) (superpixel or component):

1. Compute its centroid \(c(R)\) (may be non-integer).
2. If \(c(R)\in R\), use \(c(R)\).
3. Otherwise (rare), use the nearest pixel in \(R\) to \(c(R)\).

This avoids any “centroid must lie inside” assumptions.

---

## 4. Algorithm

### Step 0 — Preprocess
1. \(I' \leftarrow \mathrm{ResizeLongSide}(I, L)\)

### Step 1 — SLIC superpixels
2. \(S \leftarrow \mathrm{SLIC}(I', K, c)\) in RGB space
3. For each superpixel \(j\in\{1,\dots,K\}\):
   - \(\Omega_j \leftarrow \{x \mid S(x)=j\}\)
   - \(\mu_j \leftarrow c(\Omega_j)\) (with the deterministic placement rule above)
   - \(g_j \leftarrow \frac{1}{|\Omega_j|}\sum_{x\in\Omega_j} \mathrm{Gray}(I'(x))\)

### Step 2 — Initialize prompts (C fix: deterministic)
4. Define center-window superpixels:
   - Normalize centroid coordinates to \([0,1]^2\): \(\mu_j^{\text{norm}}\)
   - \(\mathcal{C} \leftarrow \{ j \mid \mu_j^{\text{norm}} \in [\alpha,1-\alpha]^2 \}\)

5. Initialize **positive** superpixels (darkest in center window):
   - Sort \(\mathcal{C}\) by grayscale ascending (smaller \(g_j\) = darker)
   - \(\mathcal{J}_+ \leftarrow\) first \(n_+\) indices
   - \(P \leftarrow \{(\mu_j, +1)\mid j\in\mathcal{J}_+\}\)

6. Define boundary superpixels:
   - \(\mathcal{B} \leftarrow \{ j \mid \Omega_j \cap \partial\Omega \neq \varnothing \}\)
     where \(\partial\Omega\) is the image boundary.

7. Initialize **negative** superpixels (brightest on boundary):
   - Sort \(\mathcal{B}\) by grayscale descending (larger \(g_j\) = brighter)
   - \(\mathcal{J}_- \leftarrow\) first \(m_-\) indices
   - \(P \leftarrow P \cup \{(\mu_j, -1)\mid j\in\mathcal{J}_-\}\)

8. Maintain used-index sets:
   - \(\texttt{PosIdx} \leftarrow \mathcal{J}_+\)
   - \(\texttt{NegIdx} \leftarrow \mathcal{J}_-\)

9. Initialize mask prompt:
   - \(M^{\text{prompt}} \leftarrow \mathbf{0}\) (empty)

10. Initialize candidate list:
   - \(\mathcal{Q} \leftarrow []\)  (store \((M_t, s_t)\))

### Step 3 — Iterative inference
For \(t = 1,2,\dots,T\):

11. Run SAM2:
   - \((\ell_t, M_t, s_t) \leftarrow \mathrm{SAM2}(I', M^{\text{prompt}}, P)\)
   - \(p_t \leftarrow \sigma(\ell_t)\)

12. Soft-vote scoring (exclude already used positive/negative):
   - For each \(j \notin (\texttt{PosIdx}\cup\texttt{NegIdx})\):
     \[
       v_j = \frac{1}{|\Omega_j|}\sum_{x\in\Omega_j} p_t(x)
     \]
   - \(j^\star \leftarrow \arg\max_{j \notin (\texttt{PosIdx}\cup\texttt{NegIdx})} v_j\)

13. Stop condition (main):
   - If \(j^\star \in \mathcal{B}\): **break**

14. Optional Top-\(q\) rerun (D fix: deterministic definition):
   - Let \(\mathcal{J}_{\text{top}}\) be the top-\(q\) indices by \(v_j\), with the same exclusion rule.
   - For each \(j\in\mathcal{J}_{\text{top}}\):
     - \((\ell^{(j)}, M^{(j)}, s^{(j)}) \leftarrow \mathrm{SAM2}(I', M^{\text{prompt}}, P \cup \{(\mu_j, +1)\})\)
   - Choose \(j^\star \leftarrow \arg\max_{j\in\mathcal{J}_{\text{top}}} s^{(j)}\)
   - Set \((M_t, s_t) \leftarrow (M^{(j^\star)}, s^{(j^\star)})\)

15. Add the chosen positive prompt:
   - \(P \leftarrow P \cup \{(\mu_{j^\star}, +1)\}\)
   - \(\texttt{PosIdx} \leftarrow \texttt{PosIdx} \cup \{j^\star\}\)

16. Optional non-convex refinement (B fix: no false convex guarantees):
   - Let \(s_0 \leftarrow \mathrm{FGRegion}(M_t)\)  (foreground region from current mask)
   - Let \(s_1 \leftarrow \mathrm{ConvHull}(s_0)\)
   - Compute area ratio:
     \[
       r = \frac{|s_0|}{|s_1|}
     \]
   - If \(r < \tau_{\text{convex}}\):
     - \(s_2 \leftarrow s_1 \setminus s_0\)
     - \(s_3 \leftarrow \mathrm{LargestCC}(s_2)\)
     - Add positive point at \(c(s_3)\):
       \[
         P \leftarrow P \cup \{(c(s_3), +1)\}
       \]

17. Optional positive-point pruning (E fix: deterministic keep policy):
   - Consider only positive points in \(P\).
   - If two positive points are within Euclidean distance \(< d_{\min}\),
     **keep the latest added one** (deterministic) and remove the others.

18. Update next mask prompt (positive superpixel union minus negative union):
   \[
     M^{\text{prompt}}
     \leftarrow
     \Big(\bigcup_{(u,+1)\in P} \Omega_{S(u)}\Big)
     \setminus
     \Big(\bigcup_{(u,-1)\in P} \Omega_{S(u)}\Big)
   \]

19. Append candidate:
   - \(\mathcal{Q}.\text{append}((M_t, s_t))\)

**End For**

---

## 5. Final mask selection from the nested sequence

Let candidates be \(\{(M_i, s_i)\}_{i=1}^{N}\) from \(\mathcal{Q}\).

### 5.1 IoU de-dup (optional but recommended)
- If \(\mathrm{IoU}(M_a, M_b) > \tau_{\text{iou}}\), keep only the one with higher score.

Let remaining candidates be \(\{(M_i, s_i)\}_{i=1}^{N'}\).

### 5.2 Fallback for small N
- If \(N' < 3\): return \(M^\star = \arg\max_i s_i\).

### 5.3 1D k-means clustering by area (k=3)
- Feature: \(a_i = |M_i|\)
- Initialize centers with \(\min(a)\), \(\mathrm{median}(a)\), \(\max(a)\)
- Run 1D k-means to convergence.

Let the three clusters be ordered by centroid area, and denote the middle one as \(C_{\text{mid}}\).

### 5.4 Default final choice (as you requested)
- Return the **largest-area mask in the middle cluster**:
  \[
    M^\star = \arg\max_{M_i \in C_{\text{mid}}} |M_i|
  \]

---

## Output
- Final mask \(M^\star\)