# 4.5-4.6 完整 GPTModel + 文本生成
import torch
import torch.nn as nn


# ============================================================
# GPT 配置
# ============================================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


# ============================================================
# 核心组件
# ============================================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T], -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        context_vec = context_vec.view(B, T, self.d_out)
        return self.out_proj(context_vec)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["drop_rate"],
            num_heads=cfg["n_heads"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


# ============================================================
# GPTModel - 完整的 GPT 模型
# ============================================================
class GPTModel(nn.Module):
    """
    完整的 GPT 模型

    架构:
    Token Embedding ──┐
                     ├──→ + → Dropout → [TransformerBlock]×N → LayerNorm → Linear → logits
    Position Embedding┘
    """
    def __init__(self, cfg):
        super().__init__()

        # 嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer 块堆叠
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 输出层（与 token embedding 共享权重）
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        # 权重共享：输出层权重 = token embedding 权重的转置
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # 1. 嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # 2. Transformer 块
        x = self.trf_blocks(x)

        # 3. 归一化 + 输出
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================================
# 文本生成
# ============================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    自回归文本生成

    Args:
        model: GPT 模型
        idx: 输入 token IDs, shape (batch_size, seq_len)
        max_new_tokens: 最大生成 token 数
        context_size: 模型最大上下文长度

    Returns:
        生成的 token IDs, shape (batch_size, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # 如果序列超过上下文长度，截断
        idx_cond = idx[:, -context_size:]

        # 前向传播
        with torch.no_grad():
            logits = model(idx_cond)

        # 只取最后一个 token 的 logits
        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # softmax → 概率分布 → 选择概率最高的 token (greedy decoding)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch_size, 1)

        # 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("4.5 GPTModel 完整测试")
    print("=" * 50)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    # 模拟输入
    batch = torch.tensor([
        [6109, 3626, 6100, 345],
        [6109, 1110, 6622, 257]
    ])
    print(f"\n输入形状: {batch.shape}")

    logits = model(batch)
    print(f"输出形状: {logits.shape}")

    # ============================================================
    print("\n" + "=" * 50)
    print("4.6 参数统计")
    print("=" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"约 {total_params / 1e6:.2f}M 参数")

    # 可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,}")

    # 内存占用 (float32 = 4 bytes per param)
    mem_mb = total_params * 4 / (1024 * 1024)
    print(f"模型权重内存: {mem_mb:.2f} MB")

    # ============================================================
    print("\n" + "=" * 50)
    print("4.7 文本生成测试")
    print("=" * 50)

    model.eval()  # 切换到评估模式（关闭dropout）

    start_context = torch.tensor([[6109, 3626, 6100, 345]])  # "Every effort moves you"
    print(f"\n输入: {start_context}")
    print(f"输入形状: {start_context.shape}")

    # 生成 10 个新 token
    generated = generate_text_simple(
        model=model,
        idx=start_context,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print(f"\n生成结果: {generated}")
    print(f"生成形状: {generated.shape}")
    print(f"生成的 token IDs: {generated[0, 4:].tolist()}")

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
