# 3.3.1 简化版自注意力机制（不含可训练权重）
import torch

# 输入：句子 "Your journey starts with one step" 已转换为3维嵌入向量
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

print("=" * 50)
print("3.3.1 简化自注意力 - 计算注意力得分")
print("=" * 50)

# 第1步：计算注意力得分（以第2个token "journey"作为查询）
query = inputs[1]  # x^2 作为查询向量
print(f"\n查询向量 (journey):\n{query}")

# 计算查询与所有输入token的点积（自注意力：使用输入token自身的嵌入向量作为查询，与其他token的嵌入向量计算相似度）
attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(f"\n注意力得分 (journey 对所有token):\n{attn_scores_2}")

# 第2步：归一化注意力得分（softmax）
# 公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"\n归一化注意力得分 (简单除法归一化):\n{attn_weights_2_tmp}")

# 使用 softmax 进行更稳定的归一化
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)  # dim=0 表示沿着第0维（行方向）计算softmax，即对单个token的所有注意力得分进行归一化
print(f"\n归一化注意力得分 (softmax):\n{attn_weights_2}")

# 验证：注意力权重之和为1
print(f"\n注意力权重之和: {attn_weights_2.sum()}")

print("\n" + "=" * 50)
print("3.3.2 为所有输入token计算注意力权重")
print("=" * 50)

# 计算所有token对的注意力得分矩阵
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
print(attn_scores)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(f"\n注意力得分矩阵:\n{attn_scores}")

# Using matrix multiplication for efficient computation: attn_scores = inputs @ inputs.T
# inputs.T is the transpose of inputs, which converts rows to columns
# This allows computing all pairwise dot products in a single operation
attn_scores_matrix = inputs @ inputs.T
print(f"\n注意力得分矩阵 (矩阵乘法): \n{attn_scores_matrix}")
print(f"结果是否一致: {torch.allclose(attn_scores, attn_scores_matrix)}")

# 对所有行应用softmax得到注意力权重矩阵
attn_weights = torch.softmax(attn_scores_matrix, dim=1)
print(f"\n注意力权重矩阵:\n{attn_weights}")

# # 验证每行之和为1
# print(f"\n每行注意力权重之和: {attn_weights.sum(dim=1)}")
