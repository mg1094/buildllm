# 4.4 TransformerBlock - 连接注意力层与前馈网络
import torch
import torch.nn as nn

# 复用第3章的多头注意力（直接内联避免导入问题）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"
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


# ============================================================
# LayerNorm 和 GELU（复用之前的实现）
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


# ============================================================
# Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    """
    Transformer 块（GPT 的核心组件）

    数据流:
    1. 输入 x
    2. LayerNorm → MultiHeadAttention → 残差连接 (x + attn_out)
    3. LayerNorm → FeedForward → 残差连接 (x + ff_out)
    4. 输出

    结构图:
         x
         │
         ├──→ LayerNorm → MultiHeadAttention → Dropout → + ← x (残差)
         │                                                    │
         │                                                    ↓
         └──→ LayerNorm → FeedForward → Dropout → + ← (残差)
                                                     │
                                                     ↓
                                                   输出
    """
    def __init__(self, cfg):
        super().__init__()

        # 多头注意力
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["drop_rate"],
            num_heads=cfg["n_heads"]
        )

        # 前馈网络
        self.ff = FeedForward(cfg)

        # 层归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        # 捷径连接 1: 残差 (LayerNorm 在注意力之前 - "Pre-LN" 架构)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)          # 多头注意力
        x = x + shortcut         # 残差连接

        # 捷径连接 2: 残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)           # 前馈网络
        x = x + shortcut         # 残差连接

        return x


# ============================================================
# 测试 TransformerBlock
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("4.4 TransformerBlock 测试")
    print("=" * 50)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)

    # 单个 TransformerBlock 测试
    block = TransformerBlock(GPT_CONFIG_124M)

    # 模拟输入
    x = torch.randn(2, 4, 768)  # batch=2, seq_len=4, emb_dim=768
    print(f"\n输入形状: {x.shape}")

    out = block(x)
    print(f"输出形状: {out.shape}")

    # 验证残差连接：输入输出形状一致
    print(f"输入输出形状是否一致: {x.shape == out.shape}")

    # 参数量
    block_params = sum(p.numel() for p in block.parameters())
    print(f"\n单个 TransformerBlock 参数量: {block_params:,}")

    # 12个块的总参数量
    total_trf_params = block_params * 12
    print(f"12个 TransformerBlock 总参数量: {total_trf_params:,}")
