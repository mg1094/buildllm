# 第6章：用于分类任务的微调
# 基于预训练 GPT 模型进行垃圾短信分类

import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到路径，以便导入 common 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.gpt_model import (
    GPTModel,
)


# ============================================================
# 6.2 准备数据集
# ============================================================

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """下载并解压 SMS 垃圾短信数据集"""
    if data_file_path.exists():
        print(f"{data_file_path} 已存在，跳过下载。")
        return
    
    print(f"正在下载数据集...")
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    
    print(f"正在解压...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"数据集已保存至: {data_file_path}")


def create_balanced_dataset(df):
    """创建平衡数据集（ham 和 spam 数量相同）"""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


def random_split(df, train_frac, validation_frac):
    """随机划分数据集：训练集/验证集/测试集"""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    return train_df, validation_df, test_df


# ============================================================
# 6.3 创建数据加载器
# ============================================================

class SpamDataset(Dataset):
    """SMS 垃圾短信数据集"""
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        
        # 对文本进行编码
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        # 确定最大长度
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        
        # 截断过长的序列
        self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
        ]
        
        # 填充到最大长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# ============================================================
# 6.5 添加分类头
# ============================================================


# ============================================================
# 6.5 添加分类头
# ============================================================

class ClassificationHead(nn.Module):
    """分类头：将 GPT 模型的输出映射到分类任务"""
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        """
        x: GPT 模型的输出 (batch_size, seq_len, emb_dim)
        返回: 分类 logits (batch_size, num_classes)
        """
        # 取最后一个 token 的输出用于分类
        x = self.dropout(x[:, -1, :])
        return self.linear(x)


class GPTClassifier(nn.Module):
    """GPT 分类器：GPT 模型 + 分类头"""
    def __init__(self, gpt_model, num_classes):
        super().__init__()
        self.gpt_model = gpt_model
        self.classification_head = ClassificationHead(
            emb_dim=gpt_model.tok_emb.embedding_dim,
            num_classes=num_classes
        )
        # 冻结 GPT 模型参数（可选）
        # self.freeze_gpt_layers()
    
    def freeze_gpt_layers(self):
        """冻结 GPT 模型的所有参数"""
        for param in self.gpt_model.parameters():
            param.requires_grad = False
    
    def unfreeze_gpt_layers(self):
        """解冻 GPT 模型的所有参数"""
        for param in self.gpt_model.parameters():
            param.requires_grad = True
    
    def forward(self, in_idx):
        # 获取 GPT 模型的隐藏状态
        gpt_output = self.gpt_model.forward_features(in_idx)  # (batch_size, seq_len, emb_dim)
        # 通过分类头得到分类 logits
        logits = self.classification_head(gpt_output)
        return logits


# ============================================================
# 6.6 计算分类损失和准确率
# ============================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的分类损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算多个 batch 的平均损失"""
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """计算分类准确率"""
    model.eval()
    correct_predictions = 0
    total_examples = 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                logits = model(input_batch)
                predicted_labels = torch.argmax(logits, dim=-1)
                
                correct_predictions += (predicted_labels == target_batch).sum().item()
                total_examples += target_batch.shape[0]
            else:
                break
    
    model.train()
    return correct_predictions / total_examples if total_examples > 0 else 0.0


# ============================================================
# 6.7 使用监督数据对模型进行微调
# ============================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型（关闭 dropout）"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss, train_acc, val_acc


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs,
                     eval_freq, eval_iter, scheduler=None):
    """
    微调分类器
    
    Args:
        model: GPTClassifier 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        eval_freq: 每隔多少个 epoch 评估一次
        eval_iter: 评估时使用的 batch 数
        scheduler: 学习率调度器（可选）
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epochs_seen = []
    tokens_seen = []
    global_step = -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            if scheduler is not None:
                scheduler.step()
        
        # 定期评估
        if (epoch + 1) % eval_freq == 0:
            train_loss, val_loss, train_acc, val_acc = evaluate_model(
                model, train_loader, val_loader, device, eval_iter
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            epochs_seen.append(epoch + 1)
            tokens_seen.append(global_step)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                  f"Train acc {train_acc:.3f}, Val acc {val_acc:.3f}")
    
    return epochs_seen, train_losses, val_losses, tokens_seen, train_accs, val_accs


# ============================================================
# 6.8 将 LLM 用于垃圾短信分类
# ============================================================

def classify_review(text, model, tokenizer, device, max_length=None):
    """
    对新文本进行分类
    
    Args:
        text: 输入文本
        model: GPTClassifier 模型
        tokenizer: 分词器
        device: 设备
        max_length: 最大序列长度
    
    Returns:
        预测类别（0 或 1）
    """
    model.eval()
    
    # 编码文本
    input_ids = tokenizer.encode(text)
    
    # 填充到最大长度
    if max_length is not None:
        input_ids = input_ids[:max_length]
        input_ids += [50256] * (max_length - len(input_ids))
    else:
        input_ids = input_ids[:1024]  # 截断到模型最大上下文长度
    
    # 转换为 tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    model.train()
    return predicted_class


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第6章：用于分类任务的微调 - 垃圾短信分类")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # 6.2 准备数据集
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.2 准备数据集")
    print("=" * 60)
    
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    
    # 加载数据集
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(f"\n原始数据集大小: {len(df)} 条")
    print(f"类别分布:\n{df['Label'].value_counts()}")
    
    # 创建平衡数据集
    balanced_df = create_balanced_dataset(df)
    print(f"\n平衡数据集大小: {len(balanced_df)} 条")
    print(f"平衡后类别分布:\n{balanced_df['Label'].value_counts()}")
    
    # 转换标签
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # 划分数据集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    print(f"\n训练集: {len(train_df)} 条")
    print(f"验证集: {len(validation_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    
    # 保存为 CSV
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    
    # ----------------------------------------------------------
    # 6.3 创建数据加载器
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.3 创建数据加载器")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    print(f"训练集最大序列长度: {train_dataset.max_length}")
    
    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"\n训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 验证批次维度
    for input_batch, target_batch in train_loader:
        pass
    print(f"\n输入批次维度: {input_batch.shape}")
    print(f"标签批次维度: {target_batch.shape}")
    
    # ----------------------------------------------------------
    # 6.4-6.5 使用预训练权重初始化模型 + 添加分类头
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.4-6.5 初始化模型和分类头")
    print("=" * 60)
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 尝试加载预训练权重
    gpt_model = GPTModel(GPT_CONFIG_124M)
    
    # 检查是否有预训练权重
    pretrained_path = "../ch05/gpt_model_pretrained.pth"
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        gpt_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("未找到预训练权重，使用随机初始化")
    
    # 创建分类器
    model = GPTClassifier(gpt_model, num_classes=2)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # ----------------------------------------------------------
    # 6.6 计算初始损失和准确率
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.6 计算初始损失和准确率（微调前）")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=1)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=1)
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=1)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=1)
    
    print(f"训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.3f}")
    print(f"验证损失: {val_loss:.3f}, 验证准确率: {val_acc:.3f}")
    model.train()
    
    # ----------------------------------------------------------
    # 6.7 使用监督数据对模型进行微调
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.7 开始微调模型")
    print("=" * 60)
    
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    num_epochs = 5
    eval_freq = 1
    eval_iter = max(1, len(train_loader) // 3)  # 使用 1/3 的 batch 进行评估
    
    start_time = time.time()
    epochs_seen, train_losses, val_losses, tokens_seen, train_accs, val_accs = train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter
    )
    elapsed = time.time() - start_time
    print(f"\n微调完成! 耗时: {elapsed:.1f} 秒")
    
    # ----------------------------------------------------------
    # 6.8 在测试集上评估
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("6.8 在测试集上评估")
    print("=" * 60)
    
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=None)
    test_acc = calc_accuracy_loader(test_loader, model, device, num_batches=None)
    print(f"测试损失: {test_loss:.3f}")
    print(f"测试准确率: {test_acc:.3f}")
    
    # 分类示例
    print("\n" + "=" * 60)
    print("分类示例")
    print("=" * 60)
    
    examples = [
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
        "Hey, are you coming to the party tonight?",
        "Congratulations! You've won a FREE iPhone. Click here to claim!",
        "Let's meet for coffee tomorrow morning."
    ]
    
    for text in examples:
        label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
        label_text = "spam" if label == 1 else "ham"
        print(f"文本: {text[:50]}...")
        print(f"预测: {label_text}\n")
    
    # 保存模型
    print("\n" + "=" * 60)
    print("保存分类器模型")
    print("=" * 60)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "max_length": train_dataset.max_length,
    }, "gpt_classifier.pth")
    print("模型已保存至: gpt_classifier.pth")
    
    # 绘制训练曲线
    print("\n" + "=" * 60)
    print("训练曲线")
    print("=" * 60)
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs_seen, train_losses, label="Train loss")
        ax1.plot(epochs_seen, val_losses, label="Validation loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title("Loss over epochs")
        
        ax2.plot(epochs_seen, train_accs, label="Train accuracy")
        ax2.plot(epochs_seen, val_accs, label="Validation accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.set_title("Accuracy over epochs")
        
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        print("训练曲线已保存至: training_curves.png")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")
    
    print("\n" + "=" * 60)
    print("第6章完成!")
    print("=" * 60)
