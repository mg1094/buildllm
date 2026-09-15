"""
推理时扩展工具函数
包含: top_p_filter, sample_next_token, generate_with_sampling,
      extract_answer, normalize_answer, self_consistency_vote, best_of_n
"""

import re
from collections import Counter

import torch


# ============================================================
# 采样：温度 + 核采样（top-p）
# ============================================================

def top_p_filter(probas, top_p):
    """
    核采样（nucleus sampling）的概率过滤

    做法：候选按概率降序排，从高到低累加，只保留「入场之前累计概率
    还没越过 top_p」的那些 —— 也就是把刚好把累计概率顶过阈值的那
    一个也带上。其余置零，再重新归一化。

    返回的 tensor 已经还原成原始顺序，可以直接送进 multinomial。
    """
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_p, order = torch.sort(probas, dim=-1, descending=True)

    # cumulative[i] = 前 i+1 个候选的累计概率
    cumulative = torch.cumsum(sorted_p, dim=-1)

    # 每个候选「入场前」的累计概率，第 0 个是 0
    before = torch.zeros_like(cumulative)
    before[..., 1:] = cumulative[..., :-1]

    keep = before < top_p
    keep[..., 0] = True          # 兜底：至少留一个，否则抽签池空了会崩

    trimmed = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    trimmed = trimmed / trimmed.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # 还原成原始顺序
    return torch.zeros_like(probas).scatter(-1, order, trimmed)


def sample_next_token(logits, temperature=1.0, top_p=None):
    """
    从模型输出的 logits 里抽下一个 token

    temperature <= 0 或 None 时退化成贪心（argmax），不引入随机性。
    logits 形状 (batch, seq, vocab)，只取最后一个位置。
    """
    logits = logits[:, -1, :]

    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probas = torch.softmax(logits / temperature, dim=-1)
    probas = top_p_filter(probas, top_p)
    return torch.multinomial(probas.cpu(), num_samples=1).to(logits.device)


def generate_with_sampling(model, idx, max_new_tokens, context_size,
                           temperature=1.0, top_p=None, eos_id=None):
    """
    带 temperature + top-p 的自回归生成

    和 common.generation.generate 的区别：多了 top-p 这一层过滤，
    并且 temperature 与 top-p 的组合逻辑集中在一个函数里。
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = sample_next_token(logits, temperature, top_p)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ============================================================
# 答案抽取与归一化
# ============================================================

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text, fallback="boxed"):
    """
    从模型输出里抠出最终答案

    fallback 决定抠不到 \\boxed{} 时怎么办：
      "boxed"    → 直接返回 None
      "number"   → 退而求其次，取最后一个数字
      "fulltext" → 再退一步，把整段文本当答案
    """
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    if fallback == "boxed":
        return None

    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1]

    if fallback == "fulltext":
        return text.strip()
    return None


def normalize_answer(answer):
    """
    把等价写法统一，避免投票时被字符串差异拆票

    例："x = 83" / "$83$" / "83.0" 都应该归一到 "83"
    """
    if answer is None:
        return None

    s = answer.strip().lower()
    s = s.replace(" ", "").replace(",", "")
    s = re.sub(r"^\$|\$$", "", s)          # 去掉包裹的 $
    s = re.sub(r"^x\s*=\s*", "", s)         # 去掉前缀 "x = "
    s = re.sub(r"\.0+$", "", s)             # 3.0 → 3
    return s or None


# ============================================================
# 多份采样：自一致性投票 / Best-of-N
# ============================================================

def _sample_answers(model, tokenizer, prompt, device, n_samples,
                    temperature, top_p, max_new_tokens, context_size, seed):
    """内部工具：抽 n 份答案，返回 (完整文本列表, 归一化短答案列表)"""
    from common.generation import text_to_token_ids, token_ids_to_text

    full_texts, short_answers = [], []
    for i in range(n_samples):
        if seed is not None:
            torch.manual_seed(seed + i)     # 每份换一个签，整体仍可复现

        idx = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate_with_sampling(
            model, idx, max_new_tokens, context_size,
            temperature=temperature, top_p=top_p,
        )
        text = token_ids_to_text(out, tokenizer)[len(prompt):]
        full_texts.append(text)
        short_answers.append(
            normalize_answer(extract_answer(text, fallback="number"))
        )
    return full_texts, short_answers


def self_consistency_vote(model, tokenizer, prompt, device,
                          n_samples=5, temperature=0.8, top_p=0.9,
                          max_new_tokens=64, context_size=256, seed=None):
    """
    自一致性投票：同一道题独立答 n 遍，取得票最多的那个答案

    平票时 final_answer 返回 None —— 宁可弃权，也不瞎猜。
    """
    full_texts, short_answers = _sample_answers(
        model, tokenizer, prompt, device, n_samples,
        temperature, top_p, max_new_tokens, context_size, seed,
    )

    counts = Counter(a for a in short_answers if a is not None)
    if not counts:
        return {"final_answer": None, "counts": {}, "answers": short_answers,
                "full_texts": full_texts, "tie": False}

    top_freq = counts.most_common(1)[0][1]
    winners = [ans for ans, freq in counts.items() if freq == top_freq]
    tied = len(winners) > 1

    return {
        "final_answer": None if tied else winners[0],
        "counts": dict(counts),
        "answers": short_answers,
        "full_texts": full_texts,
        "tie": tied,
    }


def best_of_n(model, tokenizer, prompt, device, score_fn,
              n_samples=5, temperature=0.8, top_p=0.9,
              max_new_tokens=64, context_size=256, seed=None):
    """
    Best-of-N：抽 n 份，全打分，返回分数最高的那一份

    和自一致性投票的区别：不看票数，只看分数。score_fn 接收
    完整文本，返回一个标量（越大越好）。
    """
    full_texts, short_answers = _sample_answers(
        model, tokenizer, prompt, device, n_samples,
        temperature, top_p, max_new_tokens, context_size, seed,
    )

    scores = [score_fn(t) for t in full_texts]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    return {
        "final_answer": short_answers[best_idx],
        "best_text": full_texts[best_idx],
        "best_score": scores[best_idx],
        "scores": scores,
        "full_texts": full_texts,
    }
