# 4.2 使用层归一化对激活值进行标准化
import torch
import torch.nn as nn


# ============================================================
# 演示：层归一化的原理
# ============================================================
print("=" * 50)
print("4.2.1 层归一化原理演示")
print("=" * 50)

# 创建一个示例神经网络层输出
torch.manual_seed(123)
batch_example = torch.randn(2, 5)  # 2个样本，每个5维
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

print(f"\n原始层输出:\n{out}")

# 计算均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print(f"\n均值:\n{mean}")
print(f"方差:\n{var}")


# ============================================================
# 实现 LayerNorm
# ============================================================
print("\n" + "=" * 50)
print("4.2.2 实现 LayerNorm")
print("=" * 50)


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）

    对每个样本的最后一个维度进行归一化：
    - 均值归一化为0
    - 方差归一化为1
    - 然后应用可学习的缩放(γ)和偏移(β)
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5            # 防止除以0
        self.scale = nn.Parameter(torch.ones(emb_dim))      # γ (可学习缩放)
        self.shift = nn.Parameter(torch.zeros(emb_dim))     # β (可学习偏移)

    def forward(self, x):
        # x 形状: (batch, seq_len, emb_dim) 或 (batch, emb_dim)

        # 1. 计算均值和方差 (沿最后一个维度)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 2. 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # 3. 缩放和偏移
        return self.scale * norm_x + self.shift


# ============================================================
# 测试 LayerNorm
# ============================================================
print("\n" + "=" * 50)
print("4.2.3 测试 LayerNorm")
print("=" * 50)

# 3D张量测试 (batch, seq_len, emb_dim)
torch.manual_seed(123)
x = torch.randn(2, 3, 4)  # 2个样本，3个token，4维嵌入
print(f"\n输入形状: {x.shape}")
print(f"输入:\n{x}")

ln = LayerNorm(emb_dim=4)
out = ln(x)

print(f"\n归一化输出形状: {out.shape}")
print(f"归一化输出:\n{out}")

# 验证归一化效果
out_mean = out.mean(dim=-1, keepdim=True)
out_var = out.var(dim=-1, keepdim=True, unbiased=False)
print(f"\n归一化后均值:\n{out_mean}")
print(f"归一化后方差:\n{out_var}")

# 与 PyTorch 内置 LayerNorm 对比
print("\n" + "=" * 50)
print("4.2.4 与 PyTorch 内置 LayerNorm 对比")
print("=" * 50)

torch.manual_seed(123)
x_test = torch.randn(2, 3, 4)

# 我们的实现
ln_custom = LayerNorm(emb_dim=4)
out_custom = ln_custom(x_test)

# PyTorch 内置实现
ln_torch = nn.LayerNorm(4)
out_torch = ln_torch(x_test)

print(f"自定义 LayerNorm 输出:\n{out_custom[0, 0]}")
print(f"PyTorch LayerNorm 输出:\n{out_torch[0, 0]}")
print(f"结果是否接近: {torch.allclose(out_custom, out_torch, atol=1e-5)}")
