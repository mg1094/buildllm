# 3.4 实现带有可训练权重的自注意力机制
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
print("3.4.1 逐步计算注意力权重（带可训练权重）")
print("=" * 50)

# 第1步：初始化可训练的权重矩阵 W_q, W_k, W_v
d_in = inputs.shape[1]  # 输入嵌入维度 (3)
d_out = 2  # 输出嵌入维度 (设为2以方便演示)

torch.manual_seed(123)  # 固定随机种子以便复现

W_query = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_key = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_value = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

print(f"\n输入维度: {d_in}, 输出维度: {d_out}")
print(f"W_query:\n{W_query.data}")
print(f"W_key:\n{W_key.data}")
print(f"W_value:\n{W_value.data}")

# 第2步：计算 Q, K, V 矩阵
query_2 = inputs[1] @ W_query  # 第2个token的查询向量
key_2 = inputs[1] @ W_key      # 第2个token的键向量
value_2 = inputs[1] @ W_value  # 第2个token的值向量

print(f"\nquery_2:\n{query_2}")
print(f"key_2:\n{key_2}")
print(f"value_2:\n{value_2}")

# 第3步：计算注意力得分 (Q·K)
keys = inputs @ W_key  # 所有token的键向量
attn_score_2 = query_2 @ keys.T  # query_2 与所有keys的点积
print(f"\n注意力得分: {attn_score_2}")

# 第4步：缩放注意力得分 (除以 sqrt(d_k))
d_k = keys.shape[1]  # 键向量的维度
print(f"d_k: {d_k}")
attn_score_2_scaled = attn_score_2 / (d_k ** 0.5)
print(f"缩放后的注意力得分: {attn_score_2_scaled}")

# 第5步：应用softmax得到注意力权重
attn_weights_2 = torch.softmax(attn_score_2_scaled, dim=0)
print(f"注意力权重: {attn_weights_2}")
print(f"注意力权重之和: {attn_weights_2.sum()}")

# 第6步：计算上下文向量 (注意力权重 × 值向量)
values = inputs @ W_value  # 所有token的值向量
context_vec_2 = attn_weights_2 @ values
print(f"\n上下文向量 (journey):\n{context_vec_2}")

print("\n" + "=" * 50)
print("3.4.2 实现简洁的自注意力机制 Python 类")
print("=" * 50)


class SelfAttention_v1(nn.Module):
    """自注意力机制 V1（逐步版本）"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.randn(d_in, d_out) * 0.4)
        self.W_key = nn.Parameter(torch.randn(d_in, d_out) * 0.4)
        self.W_value = nn.Parameter(torch.randn(d_in, d_out) * 0.4)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 注意力得分 (Q @ K.T)
        attn_scores = queries @ keys.T  # omega

        # 缩放
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)

        # 上下文向量
        context_vec = attn_weights @ values
        return context_vec


# 测试 V1
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
context_vecs = sa_v1(inputs)
print(f"\nSelfAttention_v1 输出:\n{context_vecs}")
print(f"输出形状: {context_vecs.shape}")


class SelfAttention_v2(nn.Module):
    """自注意力机制 V2（使用 nn.Linear，更好的初始化）"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        # nn.Linear 自动处理权重初始化和转置
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 注意力得分
        attn_scores = queries @ keys.T

        # 缩放 + softmax
        d_k = keys.shape[-1]
        # dim=-1 表示在最后一个维度（即键向量的维度，d_k）上应用softmax
        # 对于注意力得分矩阵 [seq_len, seq_len]，dim=-1 等价于 dim=1
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)

        # 上下文向量
        context_vec = attn_weights @ values
        return context_vec

# 测试 V2
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
context_vecs = sa_v2(inputs)
print(f"\nSelfAttention_v2 输出:\n{context_vecs}")
print(f"输出形状: {context_vecs.shape}")
