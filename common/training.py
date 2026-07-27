"""
训练和损失计算工具函数
包含: calc_loss_batch, calc_loss_loader, train_model_simple, evaluate_model
"""

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型（关闭 dropout）"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, text_to_token_ids, generate, tokenizer, device, start_context):
    """生成样本文本用于监控训练进度"""
    from .generation import text_to_token_ids as _text_to_token_ids
    from .generation import token_ids_to_text
    from .generation import generate as _generate
    
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = _text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = _generate(
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
    from .generation import text_to_token_ids, generate, token_ids_to_text
    
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
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size,
                temperature=1.0, top_k=50
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(f"\n[生成样本]\n{decoded_text.strip()}")
        model.train()
    
    return train_losses, val_losses, track_tokens_seen
