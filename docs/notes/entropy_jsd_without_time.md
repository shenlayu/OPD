# 无训练进度的 entropy JSD 推导

> 迁自 OPD_full_orig 的研究笔记。其中的候选目标和实现设想不代表 OPD_lite 已支持的 feature；没有为这些设想迁入实验代码。已实现功能以 [vOPD](../features/vOPD.md) 和 [rethinking](../features/rethinking.md) 为准。

固定一个 prefix，记 teacher 分布为 $p$，student 分布为 $q$。如果将 $\alpha$ 定义为 teacher 熵的确定函数，就得到一个没有显式训练时间参数的条件分布散度。以下先给出一般形式，再给出一个只保留 power 参数 $\lambda$ 的候选；这是候选目标的数学定义，尚未用于训练。

为避免混淆，$h(p)=-\sum_v p_v\log_2p_v$ 表示以 bit 为单位的 teacher 熵，$H(p)=-\sum_v p_v\ln p_v$ 表示以自然对数计算的熵。KL 和损失使用自然对数。梯度推导假设所有概率为正，符合有限 logits 的 softmax 分布。

**去掉时间不是唯一的操作。** 现有公式是 $\alpha=u^{(1+h(p))^\lambda}$。直接令 $u=1$ 会使所有位置变为 RKL；令 $u=0$ 会使所有位置变为 FKL。要保留熵的作用，需要重新指定一个时间无关的映射

$$
a(p)=g(h(p)),\qquad 0<a(p)<1.
$$

最接近旧实现的消融是把 $u$ 固定为一个常数 $u_0\in(0,1)$：

$$
a(p)=u_0^{(1+h(p))^\lambda}
=\exp\!\left[-c(1+h(p))^\lambda\right],\qquad c=-\ln u_0>0.
$$

此时 $c$ 是固定超参数，已不含训练时间。要注意 $a(p)\le u_0$；例如固定 $u_0=1/2$ 就永远不会到达靠近纯 RKL 的区域。另一种选择是在下文直接定义新的 $g$。这些选择都不是旧公式唯一推出的结果。

**一般散度及其熵表示。** 将 $a=a(p)$ 代入已有的归一化 JSD，定义

$$
m=ap+(1-a)q,
\qquad
\boxed{\displaystyle
\mathcal D_g(p,q)
=\frac{\operatorname{KL}(p\Vert m)}{1-a}
 +\frac{\operatorname{KL}(q\Vert m)}{a}.}
$$

展开两个 KL 的加权和，有

$$
\begin{aligned}
a\operatorname{KL}(p\Vert m)+(1-a)\operatorname{KL}(q\Vert m)
&=a\sum_v p_v\ln p_v+(1-a)\sum_v q_v\ln q_v-\sum_v m_v\ln m_v\\
&=H(m)-aH(p)-(1-a)H(q).
\end{aligned}
$$

所以也可以写成一个由 teacher 熵决定混合比例的 Jensen gap：

$$
\boxed{\displaystyle
\mathcal D_g(p,q)
=\frac{H\!\left(g(h(p))p+[1-g(h(p))]q\right)
-g(h(p))H(p)-[1-g(h(p))]H(q)}{g(h(p))[1-g(h(p))]}.}
$$

对于每一对 $p,q$，混合比例都是一个确定数值。由 KL 非负性，$\mathcal D_g(p,q)\ge0$；当 $0<a<1$ 时，等号要求 $p=m=q$，所以恰好在 $p=q$ 时为零。它因此是一个明确的散度。通常它不对称，也不自动满足距离的三角不等式。

**一个只保留 λ 的候选。** 为保留“低熵更接近 RKL、高熵更接近 FKL”以及“增大 $\lambda$ 更偏 FKL”的趋势，可以定义

$$
\boxed{\displaystyle a(p)=(1+h(p))^{-\lambda},\qquad\lambda>0.}
$$

这是一项新的映射选择。它使用与现有实现相同的 bit 熵尺度，但没有时间变量。对于 $h>0$，增大 $\lambda$ 严格减小 $a$；$h\to0$ 时 $a\to1$。更高的熵会使 $a$ 更接近零；固定有限词表时 $h\le\log_2|V|$，因此不会真正达到 $a=0$。

令 $b=(1+h(p))^\lambda$，则 $a=1/b$、$1-a=(b-1)/b$。由此直接得到

$$
m=\frac{p+(b-1)q}{b},
$$

$$
\boxed{\displaystyle
\mathcal D_\lambda(p,q)
=\frac{b}{b-1}\operatorname{KL}\!\left(p\middle\Vert\frac{p+(b-1)q}{b}\right)
+b\operatorname{KL}\!\left(q\middle\Vert\frac{p+(b-1)q}{b}\right),
\quad b=(1+h(p))^\lambda.}
$$

等价的熵表示为

$$
\mathcal D_\lambda(p,q)
=\frac{b^2}{b-1}H\!\left(\frac{p+(b-1)q}{b}\right)
-\frac{b}{b-1}H(p)-bH(q).
$$

沿用 $\lambda=2$ 时，$h=0.5,1,2$ bit 分别对应 $a=4/9,1/4,1/9$。$b=1$ 的边界按 $a\to1$ 的 RKL 极限处理，允许其值为无穷；实际计算也可以将 $a$ 截断至 $[10^{-3},1-10^{-3}]$，此时得到的是带截断的另一个确定散度。

**对 student 的梯度。** 给定 prefix，teacher 冻结，因此 $p$、$h(p)$ 和 $a(p)$ 对 student 参数均为常数。尽管 $m$ 依赖 $q$，两个 KL 中相关项相消，得到

$$
\frac{\partial\mathcal D_g}{\partial q_v}=\frac1a\ln\frac{q_v}{m_v}.
$$

若 $q=\operatorname{softmax}(z)$，用 softmax Jacobian 可得

$$
\boxed{\displaystyle
\frac{\partial\mathcal D_g}{\partial z_v}
=\frac{q_v}{a}\left[\ln\frac{q_v}{m_v}
-\sum_jq_j\ln\frac{q_j}{m_j}\right].}
$$

所以仍可在全词表上直接反传，不需要先改写成采样 advantage。固定 $p$ 时，概率坐标中的 Hessian 为

$$
\frac{\partial^2\mathcal D_g}{\partial q_v\partial q_j}
=\mathbf1_{v=j}\frac{p_v}{q_vm_v}\ge0.
$$

因此它对 $q$ 凸；这不意味着神经网络参数空间中的训练目标凸。上述结论也只是固定 prefix 的损失性质：实际训练会更新 student，从而改变之后采样到的 prefix 分布。

**与已有散度的关系。** 固定混合参数的 skew JSD 已有标准定义，可参见 [Nielsen 关于 generalized JSD 的论文，式 11](https://franknielsen.github.io/papers/entropy-22-00221-v3.pdf)。这里的变化是用整个 teacher 分布的熵决定混合参数，并采用端点不退化的归一化。它可以准确地称为 teacher 熵自适应的归一化 JSD；仅有这个定义，还不能据此宣称文献上的原创性。

固定 $a$ 时，令 $r=p_v/q_v$，它具有标准 $f$-divergence 表示：

$$
D_a(p,q)=\sum_vq_vf_a\!\left(\frac{p_v}{q_v}\right),
\qquad
f_a(r)=\frac{ar\ln r-(ar+1-a)\ln(ar+1-a)}{a(1-a)}.
$$

但是，当 $a=g(h(p))$ 随 $p$ 改变时，不能自动把整族散度视为由同一个固定 $f$ 生成的标准 $f$-divergence。对于上面的 $\lambda=2$ 候选，可以给出一个直接反例。任意固定 $f$ 的散度都满足

$$
D_f(p\otimes r,q\otimes r)=D_f(p,q),
$$

因为添加同一个独立随机变量不改变似然比。取 $p=(0.9,0.1)$、$q=(0.5,0.5)$、$r=(0.5,0.5)$，teacher 熵由 $0.468996$ bit 增至 $1.468996$ bit，$a$ 由 $0.463403$ 降至 $0.164044$。上述候选散度由 $0.402866$ 变为 $0.377315$，因此不具有这一不变性，也不是固定 $f$ 的散度。这个例子同时说明它不满足一般的数据处理不等式：丢弃新增的独立随机变量反而增大了散度。

**它仍然没有测量覆盖缺口。** 对同一个 teacher 分布 $p$，无论 student 已接近它，还是严重低估了 teacher 的主要模式，$a(p)$ 都相同。teacher 熵不能单独表达 student 的 mode-seeking 难度。

从梯度也能看到这一点。令 $r_v=p_v/q_v$，则

$$
\frac{\partial\mathcal D_g}{\partial q_v}
=-\frac1a\ln\left[1+a(r_v-1)\right].
$$

当 $|a(r_v-1)|\ll1$ 时，它近似 FKL 在概率单纯形上的纠错方向；当 student 极度低估 teacher token，使 $r_v$ 很大时，这个条件可能失败。对于固定 $p_v>0$ 和固定 $a>0$，$q_v\to0$ 时对应的 logit 梯度仍趋于零，而 FKL 的 logit 梯度趋于 $-p_v$。所以取消时间后，可以更干净地检验 teacher 熵这个信号本身，但没有自动解决原先关心的缺失模式问题。

训练进度不是对齐程度的直接测量，却可以承担预先设定的课程调度。固定 $u_0$ 的实验用于隔离“时间是否有帮助”，新的倒数幂映射用于检验另一种熵自适应目标；它们回答的问题不同。
