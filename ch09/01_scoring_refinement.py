# 第9章：推理时扩展（二）—— 打分与自我改进
# 从「多抽几份」走到「挑一份」和「让模型自己改一份」

import os
import sys

import math
import torch
import tiktoken

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.gpt_model import GPTModel
from common.generation import text_to_token_ids, token_ids_to_text
from common.inference import (
    generate_with_sampling,
    extract_answer,
    normalize_answer,
    best_of_n,
)


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ============================================================
# 9.1 启发式打分：看格式，看长度
# ============================================================

def format_score(text, boxed_bonus=2.0, number_bonus=1.0,
                 brevity_scale=1.5, brevity_decay=500.0):
    """
    完全不需要跑模型，纯字符串就能算出来的分数

    两个部分：
      1) 格式分 —— 有 \\boxed{} 给满，能抠出数字给一半，都没有给 0
      2) 长度分 —— 越短加分越多，按指数衰减

    这个打分器有一个必须知道的性质：它一个字都不检查答案对不对。
    它奖励的只是「长得像答案」和「够短」。
    """
    if extract_answer(text, fallback="boxed") is not None:
        score = boxed_bonus
    elif extract_answer(text, fallback="number") is not None:
        score = number_bonus
    else:
        score = 0.0

    score += brevity_scale * math.exp(-len(text) / brevity_decay)
    return score


def demo_format_score():
    """同一个答案，写得短和写得长，分数差多少"""
    print("=" * 60)
    print("9.1 启发式打分：格式 + 长度")
    print("=" * 60)

    long_answer = (
        "Let me work through this step by step. First we isolate the variable, "
        "then we divide both sides, and finally we verify by substitution. "
        r"After all that, the final result is \boxed{83}"
    )
    short_answer = r"\boxed{83}"

    for label, text in [("短答案", short_answer), ("长答案", long_answer)]:
        print(f"  {label}（{len(text)} 字符）→ 分数 {format_score(text):.3f}")

    print()
    print("  两份答案最终给的数都是 83，但短的那份分更高 ——")
    print("  这个打分器奖励的是「简洁」，不是「正确」。")
    print()


# ============================================================
# 9.2 对数概率打分：问模型自己有多确信
# ============================================================

@torch.inference_mode()
def avg_logprob_score(model, tokenizer, prompt, answer, device="cpu"):
    """
    用模型自己的对数概率给答案打分

    做法：prompt 和 answer 拼在一起喂进模型，只取「预测 answer 的
    那些位置」的对数概率，求平均。

    为什么取平均而不是求和：求和会让长答案天然吃低分，那就又变成
    一个数长度的打分器了。取平均才能跟长度解耦。
    """
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    if not answer_ids:
        return float("-inf")

    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    # 位置 t 的 logits 预测的是第 t+1 个 token，所以答案整体错开一格
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    positions = torch.arange(start, end, device=device)
    targets = full_ids[start + 1: end + 1]
    return logprobs[positions, targets].mean().item()


def demo_logprob_score(model, tokenizer, device, prompt):
    """换掉答案里的关键词，看分数怎么掉"""
    print("=" * 60)
    print("9.2 对数概率打分：问模型自己有多确信")
    print("=" * 60)
    print(f"  提示词：{prompt!r}")
    print()

    for tail in (" the sun.", " the moon.", " a teapot."):
        s = avg_logprob_score(model, tokenizer, prompt, tail, device)
        print(f"  续写 {tail!r:<14} → 平均对数概率 {s:>9.4f}")

    print()
    print("  数值都是负数，越大（越接近 0）表示模型越「顺」。")
    print("  它跟格式打分器的偏好完全不同 —— 一个看长短，一个看内容。")
    print()


# ============================================================
# 9.3 Best-of-N：全打分，选最高
# ============================================================

def demo_best_of_n(model, tokenizer, device, prompt, n_samples=5):
    """
    抽 n 份，每份打分，返回分数最高的那一份

    和自一致性投票的区别：投票看票数，Best-of-N 看分数。
    分数不够可靠的时候，票数反而是更稳的信号 —— 这是第 8 章和
    这一章对比下来最值得记住的一条。
    """
    print("=" * 60)
    print("9.3 Best-of-N：全打分，选最高")
    print("=" * 60)
    print(f"  提示词：{prompt!r}")
    print()

    result = best_of_n(
        model, tokenizer, prompt, device,
        score_fn=format_score,
        n_samples=n_samples, temperature=0.9, top_p=0.9,
        max_new_tokens=40, context_size=GPT_CONFIG_124M["context_length"],
        seed=7,
    )

    best_i = max(range(len(result["scores"])), key=lambda i: result["scores"][i])
    for i, (text, score) in enumerate(zip(result["full_texts"], result["scores"])):
        mark = "   ← 选中" if i == best_i else ""
        print(f"  [第 {i + 1} 份] {len(text):>3} 字符   分数 {score:.3f}{mark}")

    print()
    print(f"  胜出的那份（截断）：{result['best_text'][:56]!r}")
    print(f"  从里面抽出的答案：  {result['final_answer']!r}")
    print()
    print("  注意最后这行：如果模型输出里既没有 \\boxed{} 也没有数字，")
    print("  抽取就会返回 None —— 打分器给了它最高分，但它其实没给出答案。")
    print("  这正是「分数高」和「答得对」是两回事的最直接例子。")
    print()


# ============================================================
# 9.4 自我改进：draft → critique → refine
# ============================================================

CRITIQUE_TEMPLATE = (
    "Review the draft answer below. Point out any logical gaps, missing "
    "steps, or arithmetic mistakes. If it looks correct, say so briefly.\n\n"
    "Question:\n{question}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Review:"
)

REFINE_TEMPLATE = (
    "Rewrite the answer using the review below. Keep it short and end with "
    "the final result.\n\n"
    "Question:\n{question}\n\n"
    "Previous answer:\n{draft}\n\n"
    "Review:\n{critique}\n\n"
    "Revised answer:"
)


def make_critique_prompt(question, draft):
    """把草稿包成「请审阅」的提示词"""
    return CRITIQUE_TEMPLATE.format(question=question, draft=draft)


def make_refine_prompt(question, draft, critique):
    """把草稿和批评一起包成「请重写」的提示词"""
    return REFINE_TEMPLATE.format(question=question, draft=draft, critique=critique)


def self_refinement_loop(model, tokenizer, question, device,
                         rounds=2, score_fn=None, max_new_tokens=40,
                         temperature=0.7, top_p=0.9):
    """
    让模型先答一遍，再批评自己，然后带着批评重写

    收入准则只有一条：改完的分数不低于原来的才收下。这一行是整个
    循环唯一的安全带 —— 没有它，模型可能越改越差。

    返回每一步的记录，方便回头检查模型到底改了什么。
    """
    ctx = GPT_CONFIG_124M["context_length"]

    def _run(prompt):
        idx = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate_with_sampling(
            model, idx, max_new_tokens, ctx,
            temperature=temperature, top_p=top_p,
        )
        return token_ids_to_text(out, tokenizer)[len(prompt):]

    draft = _run(question)
    draft_score = score_fn(draft) if score_fn else 0.0
    steps = []

    for r in range(rounds):
        critique = _run(make_critique_prompt(question, draft))
        revised = _run(make_refine_prompt(question, draft, critique))
        revised_score = score_fn(revised) if score_fn else 0.0

        accepted = revised_score >= draft_score
        steps.append({
            "round": r + 1,
            "draft": draft,
            "critique": critique,
            "revised": revised,
            "score_before": draft_score,
            "score_after": revised_score,
            "accepted": accepted,
        })

        if accepted:
            draft, draft_score = revised, revised_score

    return {"final": draft, "final_score": draft_score, "steps": steps}


def demo_self_refinement(model, tokenizer, device, question, rounds=2):
    """跑一遍自我改进循环，把每轮的变化打出来"""
    print("=" * 60)
    print("9.4 自我改进：draft → critique → refine")
    print("=" * 60)
    print(f"  题目：{question!r}")
    print()

    result = self_refinement_loop(
        model, tokenizer, question, device,
        rounds=rounds, score_fn=format_score,
    )

    for step in result["steps"]:
        verdict = "收下" if step["accepted"] else "拒绝，保留原稿"
        print(f"  【第 {step['round']} 轮】{verdict}")
        print(f"    改写前：{step['draft']!r}  (分数 {step['score_before']:.3f})")
        print(f"    批评：  {step['critique']!r}")
        print(f"    改写后：{step['revised']!r}  (分数 {step['score_after']:.3f})")
        print()

    print(f"  最终输出：{result['final']!r}")
    print()


# ============================================================
# 9.5 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第 9 章：推理时扩展（二）—— 打分与自我改进")
    print("=" * 60)
    print()

    demo_format_score()

    device = torch.device("cpu")
    checkpoint = os.path.join(
        os.path.dirname(__file__), "..", "ch05", "gpt_model_pretrained.pth"
    )

    if not os.path.exists(checkpoint):
        print(f"找不到权重文件：{checkpoint}")
        print("跳过需要模型的演示（9.1 不依赖模型，已经跑完）")
        sys.exit(0)

    model = GPTModel(GPT_CONFIG_124M)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"模型已加载：{checkpoint}\n")

    demo_logprob_score(model, tokenizer, device, "The meaning of life is")
    demo_best_of_n(model, tokenizer, device, "The meaning of life is")
    demo_self_refinement(
        model, tokenizer, device,
        "Question: What is 12 + 7?\nAnswer:",
    )
