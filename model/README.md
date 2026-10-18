# 模型下载占位

只记录两个保留 module 用过或配置中使用的模型，不包含 MiniLLM 模型。权重未复制。
先在对应 Python 环境安装 `huggingface_hub`，然后在 OPD_lite 根目录执行各文档的命令。
已有资源也可以直接在 YAML 中填写本地绝对路径。默认启动器要求本地权重完整。

- [DeepSeek-R1-Distill-Qwen-1.5B](DeepSeek-R1-Distill-Qwen-1.5B.md)
- [JustRL-DeepSeek-1.5B](JustRL-DeepSeek-1.5B.md)
- [Olmo-3-1025-7B](Olmo-3-1025-7B.md)
- [Olmo-3-7B-Think](Olmo-3-7B-Think.md)
- [Qwen3-0.6B-Base](Qwen3-0.6B-Base.md)
- [Qwen3-1.7B](Qwen3-1.7B.md)
- [Qwen3-1.7B-Base](Qwen3-1.7B-Base.md)
- [Qwen3-4B](Qwen3-4B.md)
- [Qwen3-4B-Base](Qwen3-4B-Base.md)
- [Rethinking-Qwen3-1.7B-SFT](Rethinking-Qwen3-1.7B-SFT.md)

默认 vOPD：Qwen3-1.7B-Base → Qwen3-1.7B。
默认 rethinking：DeepSeek-R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B。
其他已用组合：Qwen3-4B-Base → Qwen3-4B、Olmo-3-1025-7B → Olmo-3-7B-Think、Rethinking-Qwen3-1.7B-SFT → Qwen3-4B。
rethinking 的 Qwen3 Base 组合可开启 `run.teacher_chat_template: true`，同时设置
`training.data.apply_chat_template_kwargs: {enable_thinking: false}`；仅派生模板和 EOS 元数据，不改原模型。
