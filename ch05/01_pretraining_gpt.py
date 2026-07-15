# 5.1-5.3 GPT 预训练：评估、训练循环和改进文本生成
import tiktoken
import torch
import torch.nn as nn
import time
import math
import os


# ============================================================
# 复用之前实现的组件
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, num_heads):
        super().__init__()
        assert d_out % num_heads == 0
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
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T], -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        context_vec = context_vec.view(B, T, self.d_out)
        return self.out_proj(context_vec)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["drop_rate"], num_heads=cfg["n_heads"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.out_head.weight = self.tok_emb.weight  # 权重共享

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


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


# ============================================================
# 文本生成函数 (基础版)
# ============================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] #获取最后一个token的logits，因为我们想要的是下一个token的预测
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tok):
    encoded = tok.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tok):
    flat = token_ids.squeeze(0)
    return tok.decode(flat.tolist())


# ============================================================
# 5.3 改进文本生成：Temperature + Top-k 采样
# ============================================================
def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    """
    改进的文本生成函数，支持：
    - Temperature scaling：控制生成的随机性
    - Top-k 采样：只从概率最高的 k 个 token 中采样
    - EOS 停止：遇到结束 token 时停止生成
    """
    for i in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 1. Top-k 采样
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 2. Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

        # 3. Softmax → 采样
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 4. EOS 停止
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================
# 5.1.3 计算训练集和验证集的损失
# ============================================================
def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),  # (batch*seq_len, vocab_size)
        target_batch.flatten()  # (batch*seq_len,)
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算多个 batch 的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
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


# ============================================================
# 5.2 训练 LLM
# ============================================================
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型（关闭 dropout）"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成样本文本用于监控训练进度"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=1.0, top_k=50
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\n[生成样本]\n{decoded_text.strip()}")
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    简单训练循环

    Args:
        model: GPT 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        eval_freq: 每隔多少个 epoch 评估一次
        eval_iter: 评估时使用的 batch 数
        start_context: 生成样本的起始文本
        tokenizer: 分词器
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}: "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 生成样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


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
