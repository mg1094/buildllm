# 第3章：实现注意力机制 - 完整实现
"""
本章实现了4种注意力机制：
1. 简化自注意力（无可训练权重）
2. 带可训练权重的自注意力
3. 因果自注意力（用于文本生成）
4. 多头自注意力（用于LLM）
"""
import torch
import torch.nn as nn


# ============================================================
# 1. 简化自注意力机制（3.3节）
# ============================================================
def simple_self_attention(inputs):
    """简化版自注意力，无训练权重，仅用于理解概念"""
    # 计算注意力得分矩阵
    attn_scores = inputs @ inputs.T
    # Softmax归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)
    # 上下文向量 = 注意力权重 @ 输入
    context_vec = attn_weights @ inputs
    return context_vec, attn_weights


# ============================================================
# 2. 带可训练权重的自注意力（3.4节）
# ============================================================
class SelfAttention(nn.Module):
    """带可训练权重的自注意力机制"""

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


# ============================================================
# 3. 因果自注意力机制（3.5节）
# ============================================================
class CausalAttention(nn.Module):
    """因果自注意力 - 屏蔽未来信息，用于文本生成"""

    def __init__(self, d_in, d_out, context_length, dropout_rate=0.0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        # 因果掩码
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 转置以支持batch矩阵乘法: (B, T, d_out) @ (B, d_out, T)
        attn_scores = queries @ keys.transpose(1, 2)
        d_k = keys.shape[-1]

        # 应用因果掩码
        mask = self.mask[:T, :T]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# ============================================================
# 4. 多头自注意力机制（3.6节）- LLM核心组件
# ============================================================
class MultiHeadAttention(nn.Module):
    """多头自注意力 - GPT类模型的核心组件"""

    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads):
        """
        Args:
            d_in: 输入嵌入维度
            d_out: 输出嵌入维度
            context_length: 最大序列长度
            dropout_rate: dropout比率
            num_heads: 注意力头数
        """
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

        # Q, K, V 投影
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 分割多头
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_scores = queries @ keys.transpose(2, 3)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)

        # 因果掩码
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T], -torch.inf)

        # Softmax + Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 + 合并头
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        context_vec = context_vec.view(B, T, self.d_out)

        # 输出投影
        context_vec = self.out_proj(context_vec)

        return context_vec


# ============================================================
# 测试所有实现
# ============================================================
if __name__ == "__main__":
    # 示例输入
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55]   # step
    ]).unsqueeze(0)  # 添加batch维度: (1, 6, 3)

    print("第3章注意力机制测试")
    print("=" * 50)

    # 1. 简化自注意力
    ctx, weights = simple_self_attention(inputs.squeeze(0))
    print(f"\n1. 简化自注意力 - 输出形状: {ctx.shape}")

    # 2. 带权重的自注意力
    torch.manual_seed(123)
    sa = SelfAttention(d_in=3, d_out=2)
    ctx = sa(inputs.squeeze(0))
    print(f"2. 自注意力(可训练) - 输出形状: {ctx.shape}")

    # 3. 因果自注意力
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout_rate=0.1)
    ctx = ca(inputs)
    print(f"3. 因果自注意力 - 输出形状: {ctx.shape}")

    # 4. 多头自注意力
    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=3, d_out=6, context_length=6, dropout_rate=0.1, num_heads=2)
    ctx = mha(inputs)
    print(f"4. 多头自注意力 - 输出形状: {ctx.shape}")

    print("\n所有注意力机制测试通过!")
