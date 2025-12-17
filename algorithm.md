## sam-tta

### 输入
- 测试图像：`I`
- `BasePipeline(·)` -> 逐步迭代前景，得到备选的mask pool，每个mask对应当时的prompt
- SAM2 预测器（student）：`f_θ`（**仅 mask decoder 挂 LoRA，可训练；其余冻结**）
- 配置参数：
  - `K`：用于 teacher 构造的 top-k mask 数
  - `Kp`BasePipeline中prompt的总步数
  - `inner_index`、`outer_index`：用于确定前景/不确定/背景的累计 mask 索引（`1 ≤ inner_index < outer_index ≤ K`）
  - `V`：每个 TTA step 的不同views的个数
  - `T`：TTA 更新步数
  - 损失权重：`λ_a, λ_e, λ_c`
  - 一致性权重：`α, β`
- 用于做图像增强的变换族：可逆几何变换 `g(·)` 与逆变换 `g^{-1}(·)`（缩放、翻转等）

### 输出
- 逐图像适配后的 LoRA 参数：`θ_T`（处理下一张图前重置为 `θ_0`）
- 最终预测分割：`S*`

---

## Step 0 — 无需TTA，先运行一次sam2的BasePipeline
1) 运行无监督基础分割：
   - `BasePipeline(Image) -> (P_{1:Kp}, M_pool)`
   - `P_{1:Kp}`：prompt 序列/历史
   - `M_pool = {(M_i, s_i)}`：mask 候选与得分
2) 冻结用于 TTA 的 prompt（不再搜索/更新）：
   - `P <- P_{Kp}`

---

## Step 1 — Teacher 构建
### 1.1 Top-K 选择
   我们通过一定的筛选指标拿到K个质量比较高的伪标签（目前选用的方法是聚类取中间的簇）
- `{(M_(1), s_(1)), …, (M_(K), s_(K))} <- TopK(M_pool, K)`  

### 1.2 由于我们利用sam2逐步扩张前景，因此可以找到嵌套的前景序列：
定义累计 mask `C_j`：
- `C_1 = M_(1)`
- `C_j = C_{j-1} ∪ M_(j),  j = 2..K`

满足嵌套关系：
- `C_1 ⊂ C_2 ⊂ … ⊂ C_K`（单调非减）

### 1.3 构造soft teacher
- s_(j)表示第j个student
- teacher模型为students的加权平均

### 1.4 参考TTA的方法：
- 对高置信度的前景用更高的权重
- 对低置信的前景用更低的权重
- 由于存在嵌套关系：`C_1 ⊂ C_2 ⊂ … ⊂ C_K`
   - 这里：C_1是确定的前景， C_k以外的区域是确定的背景
   - C_1到C_k之间的范围是不确定的前景，用更小的权重进行监督

---

## Step 2 — 初始化 TTA 优化器（对每个mini_batch进行tta,这里取batch_size=1, 则为逐图像的tta）
- 冻结除 LoRA 外所有参数
- 仅优化 decoder LoRA 参数 `θ`
- 配置优化器、学习率、梯度裁剪等超参数
- EMA teacher（作为超参数决定是否需要）

---

## Step 3 — TTA 迭代更新（t = 1..T）
对 `t = 1..T` 重复：

### 3.1 图像增广
- 采样 `{g_v}_{v=1..V}`：
  - 缩放 `scale ∈ {0.75, 1.0, 1.25}`
  - 水平翻转
  - 变换可逆，因此存在 `g_v^{-1}`

### 3.2 原视角预测（作为主分支）
- `S0 = f_θ(I, P)`

### 3.3 多视角预测并反变换回原坐标
对每个视角 `v = 1..V`：
- `I_v = g_v(I)`
- `P_v = g_v(P)`（仅几何映射）
- `S_v = f_θ(I_v, P_v)`
- `Ŝ_v = g_v^{-1}(S_v)`（warp 回原图坐标）

### 3.4 损失函数——这里到底用什么损失，要做成超参数去调一下
#### (a) anchor loss：确定前景/背景上对齐 teacher
- `L_anchor = BCE(S0, T)`，仅在 `R_fg ∪ R_bg` 上计算
#### (b) entropy最小化：仅在不确定区域
#### (c) 多视角一致性：Soft Dice 距离
定义 SoftDice：
- `SoftDice(A, B) = (2·Σ(A·B) + ε) / (ΣA + ΣB + ε)`
- `D(A, B) = 1 - SoftDice(A, B)`

一致性项：
- `L_cons = (1/V) Σ_v [ α·D(S0|R_unc, Ŝ_v|R_unc) + β·D(S0|R_fg∪R_bg, Ŝ_v|R_fg∪R_bg) ]`

#### (d) 无监督损失
- `L_t = λ_a L_anchor + λ_e L_entropy + λ_c L_cons`

#### (e) 利用伪标签的dice loss做监督损失


### 3.5 对LoRA层做一次反向传播

---

## Step 4 — 输出与重置（逐图像）
- `S* = f_θ(I, P)`（用最终 LoRA 参数得到最终预测）
- 返回：`S*` 与 LoRA 参数
- 在处理下一batch测试图像前：将 LoRA 参数重置回 `θ_0`（逐图像 TTA）