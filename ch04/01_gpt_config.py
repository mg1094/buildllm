# 4.1 实现 LLM 的架构 - GPT 配置
import torch
import torch.nn as nn


# GPT-2 124M 模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小 (BPE tokenizer)
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # Dropout 比率
    "qkv_bias": False        # QKV 是否使用偏置
}


# ============================================================
# 占位模型（Dummy Model）- 用于理解整体架构
# ============================================================

class DummyTransformerBlock(nn.Module):
    """占位 Transformer 块（后续会替换为真实实现）"""
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    """占位 LayerNorm（后续会替换为真实实现）"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    """
    占位 GPT 模型 - 展示整体架构
    数据流: 输入token → token嵌入 + 位置嵌入 → dropout → Transformer块 → LayerNorm → 输出层 → logits
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 多个 Transformer 块堆叠
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终的层归一化
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 输出层：将嵌入映射回词汇表大小
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # 1. Token 嵌入
        tok_embeds = self.tok_emb(in_idx)

        # 2. 位置嵌入 (0 到 seq_len-1)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 3. 相加 + dropout
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # 4. 通过 Transformer 块
        x = self.trf_blocks(x)

        # 5. 层归一化
        x = self.final_norm(x)

        # 6. 输出 logits (每个 token 对词汇表中每个词的概率分布)
        logits = self.out_head(x)
        return logits


# ============================================================
# 测试占位模型
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("4.1 GPT 配置和占位模型测试")
    print("=" * 50)

    # 模拟输入: 2段文本，每段4个token
    batch = torch.tensor([
        [6109, 3626, 6100, 345],    # "Every effort moves you"
        [6109, 1110, 6622, 257]     # "Every day holds a"
    ])
    print(f"\n输入形状: {batch.shape}")
    print(f"输入数据:\n{batch}")

    # 初始化模型
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    # 前向传播
    logits = model(batch)
    print(f"\n输出形状: {logits.shape}")
    # (batch_size=2, seq_len=4, vocab_size=50257)
    # 每个 token 对应词汇表中 50257 个词的 logits

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"约 {total_params / 1e6:.2f}M 参数")
