# Progressive JSD

> 迁自 OPD_full_orig 的研究笔记。其中的候选目标和实现设想不代表 OPD_lite 已支持的 feature；没有为这些设想迁入实验代码。已实现功能以 [vOPD](../features/vOPD.md) 和 [rethinking](../features/rethinking.md) 为准。

### 动机

训练早期，student 可能尚未覆盖 teacher 的重要模式。对于 teacher 认为重要、但 student 概率很低的 token，RKL 提供的直接纠错信号可能很弱，导致模式发现和收敛较慢。

**核心想法：前期更接近 FKL，先建立覆盖；后期逐步过渡到 RKL，更关注 student 当前实际生成的区域。**

### 方法

在相同的 student-generated prefix 上，记 teacher 分布为 $p$，当前 student 分布为 $q$。使用归一化的 generalized JSD：

$$
m_\alpha=\alpha p+(1-\alpha)q,
$$

$$
D_\alpha(p,q)=
\frac{\alpha\,\mathrm{KL}(p\Vert m_\alpha)+(1-\alpha)\,\mathrm{KL}(q\Vert m_\alpha)}{\alpha(1-\alpha)}.
$$

它连接两个端点：

$$
D_0=\mathrm{KL}(p\Vert q)\quad\text{（FKL）},
\qquad
D_1=\mathrm{KL}(q\Vert p)\quad\text{（RKL）}.
$$

这里 $D_0,D_1$ 按连续极限定义，实现时直接使用对应 KL。归一化用于避免原始 JSD 在两个端点退化为零。训练通过调整 $\alpha$，从偏 FKL 的目标逐步转向偏 RKL 的目标。

### 问题一：schedule 模式

#### 方案一：线性 schedule

$$
\alpha_t=\alpha_{\min}+
(\alpha_{\max}-\alpha_{\min})\frac{t}{T},
\qquad 0\le t\le T.
$$

$T$ 为调度总步数。**按训练进度逐步增大 $\alpha$**，作为最简单的基线；局限是不同 prefix 的覆盖改善速度可能不同。

#### 方案二：密度 schedule

**不只看训练步数，而是看 teacher/student 的概率比。** 当 $p_i/q_i$ 很大时，说明 student 严重低估了该 token，应保持较小的 $\alpha$。

一个初始候选规则是：

$$
R(s)=\max_i\left|\frac{p_i(s)}{q_i(s)}-1\right|,
\qquad
\alpha(s)=\min\left\{\alpha_{\max},\frac{c}{R(s)+\epsilon}\right\}.
$$

其中 $0<c<1$，$\epsilon>0$。直觉是：$\alpha|p_i/q_i-1|$ 较小时，JSD 的纠错分数更接近 FKL；错配越严重，$\alpha$ 越小，覆盖改善后才允许增大。

该规则不要求 $\alpha$ 严格单调；最大值可能过于保守，先作为候选验证。当次反向传播将算出的 $\alpha$ 视为常数。

### 问题二：topK

这里采用 mixture topK：

$$
S_\alpha=\operatorname{TopK}\!\left(\alpha p+(1-\alpha)q\right).
$$

候选集合依赖双方分布和当前 $\alpha$，不能简单用 teacher-only 或 student-only topK 替代。若一次性物化双方完整 logits，$N$ 个位置、词表大小 $V$ 对应的词表级存储为 $O(NV)$。

**问题是如何低显存地获得联合分布信息，而不是 topK 近似是否足够准确。** 两个方案都使用双方最后一层 hidden state 和各自的 unembedding：

$$
z^T=W_T h_T,\qquad z^S=W_S h_S.
$$

缓存 teacher 的 $h_T$；训练时使用当前可微的 student $h_S$。输出头共享访问，不逐位置保存。

#### 方案一：用 hidden state 和 unembedding 流式计算 topK

按词表分块重建双方 logits。第一遍计算全局 softmax 归一化常数，第二遍计算各块的 mixture 分数，并不断更新当前 topK。最终保留候选索引及概率，用于稀疏损失计算；反向也采用分块或重算。

**保留 topK 路线，但不先保存完整 logits 再筛选。** 代价是词表扫描、输出头重算和候选维护。

#### 方案二：用 hidden state 和 unembedding 直接计算 hidden-state 梯度

不选择 topK，直接分块累计完整 JSD 的梯度。记 $\ell_i=\log(q_i/m_{\alpha,i})$，$w_i^S$ 为 student 输出头第 $i$ 行对应的向量，则对 $0<\alpha<1$：

$$
\nabla_{h_S}D_\alpha
=\frac{1}{\alpha}\left[
\sum_i q_i\ell_i w_i^S
-\left(\sum_i q_i\ell_i\right)
\left(\sum_i q_iw_i^S\right)
\right].
$$

先得到全局归一化常数，再分块累计一个标量和两个 hidden 维向量，最后组合成梯度并反传 student 主干。若 student 输出头也训练，另外分块累计其参数梯度。

**直接完成全词表梯度归约，既不保存完整 logits，也不维护 topK。** 仍然需要全词表输出计算；与方案一的主要区别是取消候选选择和稀疏损失步骤。
