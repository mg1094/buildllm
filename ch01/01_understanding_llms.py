# 第1章：理解大语言模型
# LLM 基本概念、Transformer 架构和 GPT 模型概览


# ============================================================
# 1.1 LLM 是什么？
# ============================================================
print("=" * 60)
print("1.1 LLM 是什么？")
print("=" * 60)

print("""
大语言模型 (LLM) 是基于深度神经网络的模型，具备：
- 理解人类语言
- 生成连贯文本
- 解析复杂指令

'大' 指的是：
- 数十亿到数千亿的参数量
- 海量训练数据（互联网规模的文本）

核心能力：预测序列中的下一个单词
""")


# ============================================================
# 1.2 LLM 的应用
# ============================================================
print("=" * 60)
print("1.2 LLM 的应用场景")
print("=" * 60)

applications = {
    "文本生成": "撰写文章、代码、邮件等",
    "机器翻译": "英文 → 中文",
    "情感分析": "判断文本是正面/负面情绪",
    "文本摘要": "将长段落压缩为简短摘要",
    "问答系统": "根据文档回答问题",
    "聊天机器人": "OpenAI ChatGPT, Google Gemini",
    "知识检索": "从医学/法律文档中提取信息",
}

for app, desc in applications.items():
    print(f"  - {app}: {desc}")


# ============================================================
# 1.3 构建和使用 LLM 的步骤
# ============================================================
print("\n" + "=" * 60)
print("1.3 构建 LLM 的两阶段训练")
print("=" * 60)

print("""
预训练 (Pretraining)
  ├─ 使用大量无标签文本数据
  ├─ 学习目标：预测下一个单词
  └─ 产出：基础模型 (Base Model)，如 GPT-3

微调 (Fine-tuning)
  ├─ 指令微调：指令-答案对数据集
  │   └─ 例：翻译查询 → 正确翻译
  └─ 分类微调：文本-类别标签数据集
      └─ 例：邮件 → 垃圾/非垃圾
""")


# ============================================================
# 1.4 Transformer 架构
# ============================================================
print("=" * 60)
print("1.4 Transformer 架构")
print("=" * 60)

print("""
Transformer 架构（2017年论文 "Attention Is All You Need"）

原始 Transformer 包含两部分：
┌─────────────────────────────────────────┐
│              Transformer                 │
├──────────────────┬──────────────────────┤
│    编码器 (Encoder)    │    解码器 (Decoder)    │
│  - 处理输入文本      │  - 生成输出文本        │
│  - 编码为向量表示    │  - 使用编码向量生成    │
│  - 双向注意力        │  - 因果注意力（掩码）  │
└──────────────────┴──────────────────────┘

核心组件：自注意力机制 (Self-Attention)
  - 允许模型对序列中不同位置赋予不同权重
  - 捕捉长距离依赖关系
  - 解决 RNN 的遗忘问题
""")


# Transformer 层数对比
print("不同模型的 Transformer 层数对比：")
models_config = {
    "GPT-2 124M": {"layers": 12, "heads": 12, "emb_dim": 768},
    "GPT-2 355M": {"layers": 24, "heads": 16, "emb_dim": 1024},
    "GPT-2 774M": {"layers": 36, "heads": 20, "emb_dim": 1280},
    "GPT-2 1.5B":  {"layers": 48, "heads": 25, "emb_dim": 1600},
}

for name, cfg in models_config.items():
    print(f"  {name}: {cfg['layers']}层, {cfg['heads']}头, {cfg['emb_dim']}维")


# ============================================================
# 1.5 GPT 架构深入
# ============================================================
print("\n" + "=" * 60)
print("1.6 GPT 架构详解")
print("=" * 60)

print("""
GPT (Generative Pre-trained Transformer) 特点：
- 仅使用 Transformer 的解码器部分
- 专注于文本生成任务
- 因果注意力：只能看到当前 token 及之前的内容

GPT 数据流：
输入 Token IDs → Token Embedding + Position Embedding
                ↓
              Dropout
                ↓
         [Transformer Block] × N 层
                ↓
           LayerNorm
                ↓
            Linear 层
                ↓
          输出 Logits (词汇表概率)
""")


# 演示 GPT 配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE 分词器词汇表大小
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # Dropout 比率
    "qkv_bias": False        # QKV 是否使用偏置
}

print("GPT-2 124M 配置：")
for key, value in GPT_CONFIG_124M.items():
    print(f"  {key}: {value}")

# 参数量估算
total_params = GPT_CONFIG_124M["n_layers"] * (
    # 每个 Transformer 层的参数
    4 * GPT_CONFIG_124M["emb_dim"]**2  # 注意力 QKV + 输出投影
    + 8 * GPT_CONFIG_124M["emb_dim"]**2  # 前馈网络
    + 5 * GPT_CONFIG_124M["emb_dim"]  # 层归一化等
) + (
    # Embedding 和输出层
    GPT_CONFIG_124M["vocab_size"] * GPT_CONFIG_124M["emb_dim"]  # Token embedding
    + GPT_CONFIG_124M["context_length"] * GPT_CONFIG_124M["emb_dim"]  # Position embedding
    + GPT_CONFIG_124M["vocab_size"] * GPT_CONFIG_124M["emb_dim"]  # Output head
)

print(f"\n预估总参数量: {total_params / 1e6:.1f}M (约 1.24 亿)")


# ============================================================
# 1.7 本书各章节与 LLM 构建流程的对应关系
# ============================================================
print("\n" + "=" * 60)
print("1.7 本书章节与 LLM 构建流程")
print("=" * 60)

chapters = {
    "第1章": "理解大语言模型（理论介绍）",
    "第2章": "处理文本数据（分词、数据加载）",
    "第3章": "实现注意力机制（自注意力、多头注意力）",
    "第4章": "实现 GPT 模型（完整模型架构）",
    "第5章": "预训练 GPT（无监督学习）",
    "第6章": "分类任务微调（垃圾短信分类）",
    "第7章": "指令遵循微调（问答、翻译等）",
}

print("\nLLM 构建流程：")
for chapter, desc in chapters.items():
    print(f"  {chapter}: {desc}")


# ============================================================
# 1.8 本章摘要
# ============================================================
print("\n" + "=" * 60)
print("1.8 本章摘要")
print("=" * 60)

summary = """
关键要点：

1. LLM 是基于 Transformer 架构的深度神经网络
   - 在海量文本数据上预训练
   - 通过预测下一个单词学习语言模式

2. Transformer 的核心是自注意力机制
   - 捕捉序列中的长距离依赖
   - 允许并行计算，比 RNN 更高效

3. GPT 使用 Transformer 解码器部分
   - 因果注意力（只能看到历史信息）
   - 擅长文本生成和补全

4. 两阶段训练流程
   - 预训练：学习通用语言能力
   - 微调：适应特定任务

5. 本书将逐步实现：
   - 数据预处理 → 注意力机制 → GPT 模型
   - 预训练 → 分类微调 → 指令微调
"""
print(summary)

print("=" * 60)
print("下一章：处理文本数据（分词器、数据加载器、词嵌入）")
print("=" * 60)
