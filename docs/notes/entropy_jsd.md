# Entropy JSD 方法说明

> 迁自 OPD_full_orig 的研究笔记。本文描述已经实现的 entropy JSD；当前开关见 [rethinking features](../features/rethinking.md)。

Entropy JSD 在 student 生成的 prefix 上，根据 teacher 的下一 token 分布熵，为每个预测位置选择一个 JSD 参数。当前版本同时使用训练进度和 teacher 熵；power 版本将熵相关的指数进一步放大。本文对应当前全词表直接反传、温度扩展参数 β＝1 的实现。

记一个位置的 prefix 为 $s$，teacher 和 student 的下一 token 分布分别为

$$
p_v=P_T(v\mid s),\qquad q_v=P_\theta(v\mid s).
$$

同一位置的所有词表候选共用一个 $\alpha$；不同 prefix 可以有不同的 $\alpha$。这里的 teacher entropy 是整个分布的熵，不是某个采样 token 的负对数概率。

**基础损失。** 定义混合分布与归一化 generalized JSD：

$$
m_\alpha=\alpha p+(1-\alpha)q,
$$

$$
D_\alpha(p,q)
=\frac{\alpha\operatorname{KL}(p\Vert m_\alpha)
 +(1-\alpha)\operatorname{KL}(q\Vert m_\alpha)}{\alpha(1-\alpha)}
=\frac{\operatorname{KL}(p\Vert m_\alpha)}{1-\alpha}
 +\frac{\operatorname{KL}(q\Vert m_\alpha)}{\alpha}.
$$

端点按连续极限定义，实现时直接计算对应 KL：

$$
D_0(p,q)=\operatorname{KL}(p\Vert q)\quad\text{FKL},
\qquad
D_1(p,q)=\operatorname{KL}(q\Vert p)\quad\text{RKL}.
$$

因此，小 $\alpha$ 靠近 FKL，大 $\alpha$ 靠近 RKL。归一化避免未归一化 JSD 在两个端点变成零。中间的 $m_\alpha$ 依赖 student，反传时保留这条梯度路径。

**熵与训练进度。** 总共训练 $K$ 次更新，第 $k$ 次更新的进度是

$$
u_k=\frac{k-1}{K-1},\qquad k=1,\ldots,K.
$$

teacher 熵以 bit 为单位，且不除以 $\log_2|V|$：

$$
h(p)=-\sum_v p_v\log_2 p_v.
$$

原始 entropy JSD 取 $\lambda=1$；当前 power 实验取 $\lambda=2$：

$$
\widetilde\alpha_k(s)=u_k^{(1+h(p))^\lambda}.
$$

实际使用的参数为

$$
\alpha_k(s)=
\begin{cases}
0,&k=1,\\
1,&k=K,\\
\operatorname{clip}\!\left(\widetilde\alpha_k(s),10^{-3},1-10^{-3}\right),&1<k<K.
\end{cases}
$$

忽略数值截断时，在同一训练进度下，teacher 熵越高，$\alpha$ 越小；增大 $\lambda$ 也会使有正熵的位置更偏向 FKL。第一步仍是纯 FKL，最后一步仍是纯 RKL。这里没有假定 teacher 熵会随训练下降：teacher 虽然冻结，student 产生的 prefix 会改变，所访问位置的熵可以升高或降低。

**一次训练更新。** student 先采样回答；teacher 和 student 在相同 prefix 上产生全词表分布。teacher 分布与熵不参与反传，student 分布参与完整的 JSD 梯度计算。仅在有效 response 预测位置计算损失，在 microbatch 内取 token 平均，再沿用 actor 的动态 microbatch 聚合规则完成更新。当前版本不做 topK 截断，也不把损失替换成采样 token 的 advantage。

日志 `actor/jsd_alpha` 记录全局进度 $u_k$；`actor/jsd_effective_alpha` 才是有效 response 位置上、跨 microbatch 和 GPU 按 token 数加权的实际 $\alpha$ 均值。`actor/jsd_alpha_floor_fraction` 记录落在数值下界的比例。

**解释范围。** teacher 熵描述其分布的集中程度，并不直接测量 student 的覆盖缺口。因此，“熵高时更偏 FKL”是一个待验证的调度假设，不能推出高熵的位置一定更难对齐，也不能推出低熵的位置就适合 RKL。

实现对应 [objective.py](../../modules/rethinking/objective.py) 中的 `entropy_alpha`、`token_divergence`，以及 [dense_jsd.py](../../modules/rethinking/verl/workers/actor/dense_jsd.py) 中的 `update_dense_jsd`。去掉训练进度后的候选定义和推导见 [无训练进度的 entropy JSD](entropy_jsd_without_time.md)。
