1. 学习率 不同学习率 + cosine annealing / linear annealing
2. 改散度 
    - $\alpha$ FKL + $(1-\alpha)$ RKL
    - $\alpha$ 散度
3. 不清楚时间步信息是否有价值
    - 尝试去掉时间步信息，直接将 token entropy 代入得到一个新的散度
4. 在已有的结果基础上评测一些其它 benchmark (比如 math500, minerva，探究为什么不同 benchmark 的结果趋势不同)
5. 在已有的两个 feature （Entropy power 和 dense_beta_max）基础上做一些调参或扩展
6. 设计其它 schedule 方式（例如同时依赖 student 和 teacher）
7. 找一些其它 codebase, 尝试在其上效果与 RKL 比较
8. 其它模型组合（比如 Qwen3 1.7B 蒸 4B 之类的）

其它，任何方向都可以尝试


做实验的时候可以优先用 rethinking 那个 codebase, Qwen3 1.7B Base 蒸 Qwen3 1.7B Instruct 组合，**可以只训练 40 steps 对照结果**
做完实验之后可以填一下 docs/experiment_results.md, 在完成人那里标记自己的名字。如果对实验结果分析有什么发现可以在 docs/findings 写一下文档（任何微小的分析都可以）