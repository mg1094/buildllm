# 5.1-5.3 GPT 预训练：评估、训练循环和改进文本生成
import tiktoken
import torch
import time
import os
import sys

# 添加项目根目录到路径，以便导入 common 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.gpt_model import (
    LayerNorm,
    GELU,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    GPTModel,
)
from common.generation import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    generate,
)
from common.training import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    train_model_simple,
)


# ============================================================
# GPT 配置 (缩短上下文长度以便快速训练)
# ============================================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,     # 从1024缩短到256
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


# ============================================================
# DataLoader (复用第2章)
# ============================================================
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
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
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


# generate_text 是 generate 的别名（保持向后兼容）
generate_text = generate


# ============================================================
# DataLoader (复用第2章)
# ============================================================

# ============================================================
# 5.4 保存和加载模型权重
# ============================================================
def save_model_and_optimizer(model, optimizer, filepath):
    """保存模型和优化器状态"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filepath)
    print(f"✅ 模型已保存到: {filepath}")


def load_model_and_optimizer(model, optimizer, filepath, device):
    """加载模型和优化器状态"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"✅ 模型已从 {filepath} 加载")
    return model, optimizer


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("5. 在无标记数据集上进行预训练")
    print("=" * 60)

    # 加载分词器和数据
    tokenizer = tiktoken.get_encoding("gpt2")
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"\n文本长度: {len(raw_text)} 字符")

    # 划分训练集和验证集 (90/10)
    train_ratio = 0.90
    split_idx = int(len(raw_text) * train_ratio)
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    print(f"训练文本: {len(train_text)} 字符")
    print(f"验证文本: {len(val_text)} 字符")

    # 创建数据加载器 (使用较小的 max_length 确保有足够 batch)
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_text, batch_size=2, max_length=32,
        stride=32, shuffle=True
    )
    val_loader = create_dataloader_v1(
        val_text, batch_size=2, max_length=32,
        stride=16,  # 更小的 stride 确保验证集有足够 batch
        shuffle=False
    )

    print(f"训练 batch 数: {len(train_loader)}")
    print(f"验证 batch 数: {len(val_loader)}")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    model.eval()

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 5.1.1 测试基础文本生成（未训练）
    print("\n" + "=" * 60)
    print("5.1.1 未训练模型文本生成")
    print("=" * 60)

    start_context = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print(f"输入: {start_context}")
    print(f"输出: {token_ids_to_text(token_ids, tokenizer)}")

    # 5.1.2 计算损失
    print("\n" + "=" * 60)
    print("5.1.2 计算初始损失（未训练）")
    print("=" * 60)

    train_loss = calc_loss_loader(train_loader, model, device, num_batches=1)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=1)
    print(f"训练损失: {train_loss:.3f}")
    print(f"验证损失: {val_loss:.3f}")

    # 5.2 训练模型
    print("\n" + "=" * 60)
    print("5.2 开始训练 GPT 模型")
    print("=" * 60)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    eval_freq = 5
    eval_iter = 1

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer
    )
    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed:.1f} 秒")

    # 保存模型
    print("\n" + "=" * 60)
    print("5.4 保存模型权重")
    print("=" * 60)

    save_model_and_optimizer(model, optimizer, "gpt_model_pretrained.pth")

    # 加载模型测试
    model_new = GPTModel(GPT_CONFIG_124M)
    optimizer_new = torch.optim.AdamW(model_new.parameters())
    model_new, optimizer_new = load_model_and_optimizer(
        model_new, optimizer_new, "gpt_model_pretrained.pth", device
    )

    # 生成测试
    print("\n" + "=" * 60)
    print("加载后模型生成测试")
    print("=" * 60)

    model_new.eval()
    token_ids = generate_text(
        model=model_new,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=30,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.0,
        top_k=50
    )
    print(f"输入: {start_context}")
    print(f"输出: {token_ids_to_text(token_ids, tokenizer)}")

    # 不同 temperature 测试
    print("\n" + "=" * 60)
    print("不同 Temperature 和 Top-k 对比")
    print("=" * 60)

    for temp in [0.1, 0.5, 1.0, 2.0]:
        token_ids = generate_text(
            model=model_new,
            idx=text_to_token_ids(start_context, tokenizer).to(device),
            max_new_tokens=20,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=temp, top_k=50
        )
        out = token_ids_to_text(token_ids, tokenizer)
        print(f"  temp={temp}: {out[:60]}...")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
