# 4.3 实现带有 GELU 激活函数的前馈神经网络
import torch
import torch.nn as nn


# ============================================================
# GELU 激活函数
# ============================================================
print("=" * 50)
print("4.3.1 GELU 激活函数")
print("=" * 50)


def gelu(x):
    """
    GELU (Gaussian Error Linear Unit) 激活函数

    公式: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    特点:
    - 类似 ReLU，但在负值区域有平滑的过渡
    - 比 ReLU 更平滑，有助于训练稳定性
    - 在 Transformer 中广泛使用
    """
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
    ))


# 可视化 GELU
x = torch.linspace(-3, 3, 100)
y = gelu(x)

print(f"\nGELU 输入范围: [{x.min():.1f}, {x.max():.1f}]")
print(f"GELU 输出范围: [{y.min():.2f}, {y.max():.2f}]")

# 对比 ReLU 和 GELU
relu = nn.ReLU()
y_relu = relu(x)

print(f"\nReLU vs GELU 在负值区域:")
print(f"  x = -1.0: ReLU={relu(torch.tensor(-1.0)):.2f}, GELU={gelu(torch.tensor(-1.0)):.2f}")
print(f"  x = -0.5: ReLU={relu(torch.tensor(-0.5)):.2f}, GELU={gelu(torch.tensor(-0.5)):.2f}")
print(f"  x =  0.0: ReLU={relu(torch.tensor(0.0)):.2f}, GELU={gelu(torch.tensor(0.0)):.2f}")
print(f"  x =  0.5: ReLU={relu(torch.tensor(0.5)):.2f}, GELU={gelu(torch.tensor(0.5)):.2f}")
print(f"  x =  1.0: ReLU={relu(torch.tensor(1.0)):.2f}, GELU={gelu(torch.tensor(1.0)):.2f}")


# ============================================================
# 前馈神经网络 (FeedForward)
# ============================================================
print("\n" + "=" * 50)
print("4.3.2 实现前馈神经网络")
print("=" * 50)


class FeedForward(nn.Module):
    """
    前馈神经网络（用于 Transformer 块）

    结构: Linear → GELU → Linear
    - 第一个线性层将维度从 emb_dim 扩展到 hidden_dim (通常是 4*emb_dim)
    - GELU 激活函数
    - 第二个线性层将维度从 hidden_dim 压缩回 emb_dim
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展
            GELU(),                                         # 激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压缩
        )

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    """GELU 激活函数的 nn.Module 包装"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


# ============================================================
# 测试 FeedForward
# ============================================================
print("\n" + "=" * 50)
print("4.3.3 测试 FeedForward")
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

# 测试 GELU
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nGELU 测试:")
print(f"  输入: {x}")
print(f"  输出: {gelu(x)}")

# 测试 FeedForward
torch.manual_seed(123)
ff = FeedForward(GPT_CONFIG_124M)

# 模拟注意力层的输出
x = torch.randn(2, 4, 768)  # batch=2, seq_len=4, emb_dim=768
print(f"\nFeedForward 测试:")
print(f"  输入形状: {x.shape}")

out = ff(x)
print(f"  输出形状: {out.shape}")  # 应该和输入一样 (2, 4, 768)

# 参数量
ff_params = sum(p.numel() for p in ff.parameters())
print(f"  FeedForward 参数量: {ff_params:,}")
# Linear(768→3072) + Linear(3072→768) = 768*3072 + 3072*768 = 4,718,592
