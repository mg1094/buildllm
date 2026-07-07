# 3.5 使用因果注意力机制来屏蔽后续词
import torch
import torch.nn as nn

# 输入：与之前相同
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

print("=" * 50)
print("3.5.1 应用因果注意力掩码")
print("=" * 50)

d_in = inputs.shape[1]
d_out = 2

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa = SelfAttention_v2(d_in, d_out)

# 获取注意力权重（修改forward返回attn_weights用于调试）
keys = sa.W_key(inputs)
queries = sa.W_query(inputs)
values = sa.W_value(inputs)
attn_scores = queries @ keys.T
d_k = keys.shape[-1]

print(f"\n原始注意力得分:\n{attn_scores}")

# 创建因果掩码 (上三角为 -inf)
T = inputs.shape[0]  # 序列长度
print(f"序列长度: {T}")
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
print(f"\n因果掩码 (True表示需要遮掩的位置):\n{causal_mask}")

# 应用掩码：将未来位置的注意力得分设为 -inf
masked_attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)
print(f"\n掩码后的注意力得分:\n{masked_attn_scores}")

# 应用softmax：-inf 会变成 0
causal_attn_weights = torch.softmax(masked_attn_scores / (d_k ** 0.5), dim=-1)
print(f"\n因果注意力权重:\n{causal_attn_weights}")

# 计算上下文向量
context_vecs = causal_attn_weights @ values
print(f"\n因果上下文向量:\n{context_vecs}")

print("\n" + "=" * 50)
print("3.5.2 使用 dropout 遮掩额外的注意力权重")
print("=" * 50)

# Dropout 在训练时随机将部分注意力权重设为0，防止过拟合
# 注意：在推理时 dropout 不会生效
torch.manual_seed(123)
dropout = nn.Dropout(0.1)  # 10% dropout率
dropout_attn_weights = dropout(causal_attn_weights)
print(f"\nDropout后的注意力权重:\n{dropout_attn_weights}")

# 使用 scaled_dot_product_attention 时，dropout会在softmax后、加权求和前应用
# 为了保持注意力权重之和为1，PyTorch会在dropout时进行缩放

print("\n" + "=" * 50)
print("3.5.3 实现简洁的因果注意力类")
print("=" * 50)


class CausalAttention(nn.Module):
    """因果自注意力机制（用于GPT类模型）"""

    def __init__(self, d_in, d_out, dropout_rate=0.0):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]

        # 应用因果掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # softmax + dropout
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec, attn_weights


# 测试因果注意力
torch.manual_seed(123)
T = inputs.shape[0]
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

causal_attn = CausalAttention(d_in, d_out, dropout_rate=0.1)
context_vecs, attn_weights = causal_attn(inputs, mask=causal_mask)

print(f"\n因果注意力权重 (带dropout):\n{attn_weights}")
print(f"\n因果上下文向量:\n{context_vecs}")
print(f"输出形状: {context_vecs.shape}")

# 验证因果性：每一行只关注自身及之前的token
print(f"\n验证因果性 (第0行应只关注第0个token):")
print(f"  注意力权重第0行: {attn_weights[0].detach()}")
print(f"  第0行之和: {attn_weights[0].detach().sum()}")
