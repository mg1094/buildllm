"""
文本生成工具函数
包含: text_to_token_ids, token_ids_to_text, generate_text_simple, generate
"""

import torch


def text_to_token_ids(text, tokenizer):
    """将文本转换为 token IDs"""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将 token IDs 转换为文本"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """简单的自回归文本生成（贪心解码）"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    改进的自回归文本生成，支持 temperature、top-k 采样和 EOS 停止
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # Top-k 采样
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
        
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # EOS 停止
        if eos_id is not None and idx_next.item() == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
