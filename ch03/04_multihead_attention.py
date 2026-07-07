# 3.6 从单头注意力扩展到多头注意力
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


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads):
        """
        Args:
            d_in: 输入嵌入维度
            d_out: 输出嵌入维度
            context_length: 上下文长度（序列最大长度）
            dropout_rate: dropout比率
            num_heads: 注意力头数
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 线性层：将输入投影到 Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)  # 多头输出投影
        self.dropout = nn.Dropout(dropout_rate)

        # 注册因果掩码 buffer（不会作为模型参数保存）
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )


    def forward(self, x):
        B, T, d_in = x.shape  # batch_size, seq_len, d_in

        # 计算 Q, K, V (形状: B x T x d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 分割多头 (形状: B x T x num_heads x head_dim)
        keys = keys.view(B, T, self.num_heads, self.head_dim)
        queries = queries.view(B, T, self.num_heads, self.head_dim)
        values = values.view(B, T, self.num_heads, self.head_dim)

        # 转置以便计算注意力 (形状: B x num_heads x T x head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力得分 (形状: B x num_heads x T x T)
        attn_scores = queries @ keys.transpose(2, 3)

        # 缩放
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)

        # 应用因果掩码
        # 掩码形状: T x T，需要适配 B x num_heads x T x T
        causal_mask = self.mask[:T, :T]
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

        # Softmax + Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 (形状: B x num_heads x T x head_dim)
        context_vec = (attn_weights @ values)

        # 合并多头输出 (形状: B x T x d_out)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(B, T, self.d_out)

        # 输出投影
        context_vec = self.out_proj(context_vec)

        return context_vec, attn_weights

print("=" * 50)
print("3.6.2 多头注意力机制测试")
print("=" * 50)

# 创建示例输入 (batch_size=2, seq_len=4, d_in=3)
batch = torch.stack([inputs, inputs])  # 2个相同的样本
print(f"\n输入形状: {batch.shape}")

# 创建多头注意力
torch.manual_seed(123)
context_length = 6  # 最大序列长度
d_in = 3
d_out = 6
num_heads = 2  # 2个头，每个头维度=3
dropout_rate = 0.1

mha = MultiHeadAttention(d_in, d_out, context_length, dropout_rate, num_heads)

print(f"\n配置:")
print(f"  输入维度: {d_in}")
print(f"  输出维度: {d_out}")
print(f"  头数: {num_heads}")
print(f"  每头维度: {d_out // num_heads}")
print(f"  Dropout: {dropout_rate}")

# 前向传播
context_vecs, attn_weights = mha(batch)

print(f"\n上下文向量形状: {context_vecs.shape}")
print(f"\n注意力权重形状: {attn_weights.shape}")
print(f"  (batch_size={attn_weights.shape[0]}, num_heads={attn_weights.shape[1]}, "
      f"seq_len={attn_weights.shape[2]}, seq_len={attn_weights.shape[3]})")

print(f"\n第1个样本的注意力权重 (第1个头):\n{attn_weights[0, 0].detach()}")
print(f"\n第1个样本的上下文向量:\n{context_vecs[0].detach()}")

# 验证因果性
print("\n" + "=" * 50)
print("验证因果性")
print("=" * 50)
attn_sample = attn_weights[0, 0].detach()  # 第1个样本，第1个头
print(f"\n注意力权重 (sample=0, head=0):\n{attn_sample}")

# 检查上三角是否为0
upper_triangular = torch.triu(attn_sample, diagonal=1)
print(f"\n上三角部分 (应全为0):\n{upper_triangular}")
print(f"上三角之和: {upper_triangular.sum()} (应为0)")
