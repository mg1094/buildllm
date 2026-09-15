# buildllm

从零实现一个 GPT —— 分词、注意力、模型结构、预训练、微调，一路做到推理时扩展。

---

## 这个仓库是什么

我把大模型拆成一条主线，一个环节一个环节自己写一遍。

动机很直接：调 API 和读论文都能让人「感觉自己懂了」，但只要动手写一遍 attention 的 mask、自己推一遍 KV 的维度、自己跑一次训练循环看 loss 怎么掉，之前模糊的地方会立刻现形。

所以这里没有调库、没有 `transformers`、没有 `peft`。**每个组件都是手写的 PyTorch**，跑得起来、能验证、出问题能定位到具体哪一行。

模型规模定在 124M 量级（`emb_dim=768`、12 层、12 头），`context_length` 压到 256。选这个尺寸不是为了效果，是为了**能在 CPU 上跑完**——训练一轮几分钟，改一行立刻能看到结果，这才是学习该有的反馈速度。

---

## 进度

| 章节 | 主题 | 主要文件 | 代码量 |
|---|---|---|---|
| ch01 | 理解大语言模型 | `01_understanding_llms.py` | 223 |
| ch02 | 文本处理：分词、嵌入、位置编码 | `01_tokenizer.py`、`02_dataloader_embedding.py` | 346 |
| ch03 | 注意力机制：从简化版到多头 | 5 个文件 + `attention.py` | 681 |
| ch04 | GPT 模型结构 | 5 个文件（配置 → LayerNorm → FFN → Block → Model） | 802 |
| ch05 | 预训练 | `01_pretraining_gpt.py` | 253 |
| ch06 | 分类微调（垃圾短信识别） | `01_finetune_classification.py` | 610 |
| ch07 | 指令微调（Alpaca 风格数据） | `01_instruction_finetuning.py` | 549 |
| ch08 | 推理时扩展（一）：采样策略与自一致性 | `01_inference_scaling.py` | 281 |
| ch09 | 推理时扩展（二）：打分与自我改进 | `01_scoring_refinement.py` | 316 |

> ch08 / ch09 走的是「不改权重只改用法」这条线：同一个模型，靠调解码参数、多份采样投票、打分筛选、自我改进循环，把输出质量往上抬。这条线不需要重新训练，也不动模型结构。

---

## 目录结构

```
buildllm/
├── common/                  # 跨章节复用的模块
│   ├── gpt_model.py         # LayerNorm / GELU / MultiHeadAttention / FeedForward / TransformerBlock / GPTModel
│   ├── generation.py        # text_to_token_ids / token_ids_to_text / generate_text_simple / generate
│   ├── training.py          # 损失计算、评估、训练循环
│   └── inference.py         # top_p_filter / generate_with_sampling / 答案抽取 / 投票 / Best-of-N
│
├── ch01/ ... ch09/          # 每一章的脚本
├── requirements.txt
└── README.md
```

**为什么把公共部分抽出来**：前三章每个脚本都是自包含的，写到第 4 章开始出现大量重复——`GPTModel` 的定义在好几个文件里各抄一份，改一处就要改五处。抽到 `common/` 之后，新章节直接 `from common.gpt_model import GPTModel`，重心就能放在当章真正要讲的东西上。

---

## 环境准备

```bash
git clone https://github.com/mg1094/buildllm.git
cd buildllm

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

依赖很轻：`torch`（CPU 版）、`tiktoken`、`numpy`。没有 GPU 也能全程跑完。

---

## 怎么跑

每个脚本都能单独运行，也都往项目根目录注册了 `common` 的导入路径，所以在任何位置都能直接执行：

```bash
python ch02/01_tokenizer.py
python ch04/05_gpt_model.py
python ch05/01_pretraining_gpt.py
```

从第 5 章开始需要训练或加载权重，脚本会自动处理：

- **ch05** 跑一遍预训练，产出 `ch05/gpt_model_pretrained.pth`
- **ch06 / ch07** 各自加载上一章的权重继续微调
- **ch08 / ch09** 只做推理，直接读 `ch05/gpt_model_pretrained.pth`

> `ch08` / `ch09` 里有几节不需要模型（温度分布、top-p 过滤、答案归一化），即使权重文件不存在也会照常跑完，会打印一行提示然后跳过需要模型的演示。

---

## 各章在做什么

**ch01 — 理解大语言模型**
先不写代码，把概念摆清楚：LLM 到底在做什么、GPT 的结构长什么样、为什么是「预测下一个 token」。这一章是后面所有实现的地图。

**ch02 — 文本处理**
文本要先变成数字才能进模型。这里实现分词（BPE + 特殊 token）、Token ID 映射、滑动窗口采样造训练对、词嵌入查表、位置编码。

**ch03 — 注意力机制**
分四步走，每一步都在上一步基础上加东西：

1. 简化版自注意力（不含可训练权重）—— 先把「加权求和」这个直觉建立起来
2. 加上 `W_query` / `W_key` / `W_value` —— 变成真正能学的东西
3. 加因果 mask，防止看到未来 —— 这一步不做，模型就是作弊
4. 拆成多头 —— 让不同头关注不同的模式

最后整合成 `attention.py` 里的 `MultiHeadAttention`。

**ch04 — GPT 模型结构**
自底向上搭：配置字典 → LayerNorm → GELU + 前馈网络 → TransformerBlock（注意力 + 前馈 + 两处残差）→ 完整的 `GPTModel`。

**ch05 — 预训练**
写训练循环、算交叉熵、看 train/val loss 怎么走。跑完得到一个会接着往下写字的模型——虽然它写的东西经常重复循环，但那是真的自己训出来的。

**ch06 — 分类微调**
把预训练模型改造成垃圾短信分类器。重点在「改造」：换掉输出头、冻结部分层、用分类数据重新训练。

**ch07 — 指令微调**
用 Alpaca 风格的数据（`### Instruction` / `### Input` / `### Response`）做监督微调，让模型学会「按指令回答」而不是「接着往下写」。

**ch08 — 推理时扩展（一）**
不改权重，只改用法。温度管「票怎么分」，top-p 管「几个人参选」，自一致性投票管「多问几遍取多数」。这一章还会处理一个很容易被忽略的细节：**投票前先把答案格式归一**，否则 `'x = 83'` 和 `'83'` 会被算成两票。

**ch09 — 推理时扩展（二）**
从「多抽几份」走到「挑一份」和「让模型自己改一份」。实现两种打分器（纯字符串的格式分 vs 需要跑一遍模型的对数概率分）、Best-of-N 选择，以及 draft → critique → refine 的自我改进循环。

---

## 一些实现上的选择

**`context_length` 用 256 而不是 1024**
原始配置是 1024，但那样 CPU 上训练一轮要等太久。压到 256 之后训练几分钟一轮，改代码能立刻验证。代价是模型记不住太长的上下文——学习阶段这个取舍是划算的。

**答案抽取单独抽了一层**
`extract_answer` 支持三级降级：先找 `\boxed{}`，找不到退而求其次找数字，再找不到把整段文本当答案。看着琐碎，但它直接决定投票有没有意义——同一个答案的两种写法如果没归一，模型答对五次也可能凑不出多数。

**`common/` 只放跨章复用的东西**
章节特有的逻辑（比如 ch03 的 `attention.py`、ch09 的自我改进循环）留在章节里，不往上抽。`common/` 保持薄，读的人才能一眼看完。

---

## 权重和数据文件

`.gitignore` 里排除了 `*.pth`、`*.json`、`*.csv`、`*.png` 这类文件，所以克隆下来只有代码。权重和数据集在本地跑脚本时会自动生成或下载：

| 文件 | 来源 |
|---|---|
| `ch05/gpt_model_pretrained.pth` | 跑 `ch05/01_pretraining_gpt.py` 生成 |
| `ch05/the-verdict.txt` | 预训练语料（已入库，很短） |
| `ch06/gpt_classifier.pth` | 跑 ch06 生成 |
| `ch07/gpt_instruction_finetuned.pth` | 跑 ch07 生成 |
| `ch06/*.csv`、`ch07/*.json` | 脚本运行时自动下载 |

---

## 后续计划

推理时扩展这条线还会继续补：

- KV 缓存与流式生成 —— 目前 `MultiHeadAttention` 每次都要重算全部 K/V
- 推理预算控制：什么时候该让模型「想够了就收」
- 更大上下文下的长文本策略

再往后考虑把这些方法接到检索和工具调用上，看它们在一个真正会用的场景里表现如何。

---

*陆续更新中。*
