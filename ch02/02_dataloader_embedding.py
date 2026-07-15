# 2.6 滑动窗口采样 + 2.7 词嵌入 + 2.8 位置编码
import re
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 2.5 BPE 分词器 (tiktoken)
# ============================================================
print("=" * 50)
print("2.5 加载 tiktoken BPE 分词器")
print("=" * 50)

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print("✅ tiktoken gpt2 分词器加载成功")


# ============================================================
# 2.6 滑动窗口数据采样
# ============================================================
print("\n" + "=" * 50)
print("2.6 滑动窗口采样")
print("=" * 50)

# 读取文本
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f"\n文本总字符数: {len(raw_text)}")

# 编码整个文本
enc_text = tokenizer.encode(raw_text)
print(f"总 token 数: {len(enc_text)}")


class GPTDatasetV1(Dataset):
    """
    GPT 数据集 - 使用滑动窗口采样

    对于序列 x = [x_1, x_2, ..., x_n]，
    输入: [x_1, x_2, ..., x_{n-1}]
    目标: [x_2, x_3, ..., x_n]  (每个 token 预测下一个)
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 将整个文本编码为 token IDs
        token_ids = tokenizer.encode(txt)

        # 滑动窗口采样
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """创建 GPT 数据加载器"""
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# 测试数据加载
print("\n测试: batch_size=1, max_length=4, stride=1")
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"输入: {inputs}")
print(f"目标: {targets}")

# 解码看看
print(f"\n输入文本: {tokenizer.decode(inputs.tolist()[0])}")
print(f"目标文本: {tokenizer.decode(targets.tolist()[0])}")

# 更大 batch
print("\n" + "=" * 50)
print("测试: batch_size=8, max_length=4, stride=4")
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
inputs, targets = next(iter(dataloader))
print(f"输入形状: {inputs.shape}")  # (8, 4)
print(f"目标形状: {targets.shape}")  # (8, 4)
print(f"\n输入:\n{inputs}")
print(f"\n目标:\n{targets}")


# ============================================================
# 2.7 构建词嵌入层
# ============================================================
print("\n" + "=" * 50)
print("2.7 词嵌入层")
print("=" * 50)

# Token ID → Embedding
vocab_size = 50257  # GPT-2 词汇表大小
output_dim = 256    # 嵌入维度 (示例)

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 模拟输入: batch_size=2, seq_len=4
max_length = 4
inputs_batch = torch.tensor([[6109, 3626, 6100, 345],
                              [6109, 1110, 6622, 257]])

# 获取嵌入
embeds = token_embedding_layer(inputs_batch)
print(f"\n输入形状: {inputs_batch.shape}")
print(f"嵌入形状: {embeds.shape}")  # (2, 4, 256)
print(f"每个 token 对应一个 256 维向量")


# ============================================================
# 2.8 位置编码 (绝对位置编码)
# ============================================================
print("\n" + "=" * 50)
print("2.8 位置编码")
print("=" * 50)

context_length = 1024  # GPT-2 最大上下文长度
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 生成位置 ID
pos_ids = torch.arange(max_length)  # [0, 1, 2, 3]
pos_embeds = pos_embedding_layer(pos_ids)
print(f"位置 ID: {pos_ids}")
print(f"位置编码形状: {pos_embeds.shape}")  # (4, 256)

# 总嵌入 = Token嵌入 + 位置嵌入
total_embeds = embeds + pos_embeds
print(f"\n总嵌入形状: {total_embeds.shape}")  # (2, 4, 256)
print("每个 token 的嵌入 = token语义 + 位置信息")
