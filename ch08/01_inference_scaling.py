# 第8章：推理时扩展（一）—— 采样策略与自一致性
# 不改任何权重，只调解码参数和采样份数，看同一个模型能玩出多少花样

import os
import sys

import torch
import tiktoken

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.gpt_model import GPTModel
from common.generation import text_to_token_ids, token_ids_to_text
from common.inference import (
    top_p_filter,
    sample_next_token,
    generate_with_sampling,
    extract_answer,
    normalize_answer,
    self_consistency_vote,
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
# 8.0 加载模型
# ============================================================

def load_model(checkpoint_path, cfg=GPT_CONFIG_124M, device="cpu"):
    """加载预训练权重，返回 (model, tokenizer)"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到权重文件：{checkpoint_path}\n"
            f"请先运行第 5 章的预训练脚本，或把路径改成你自己的权重。"
        )

    model = GPTModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # 兼容两种保存格式：裸 state_dict，或者带 "model_state_dict" 的字典
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"模型已加载：{checkpoint_path}")
    return model, tokenizer


# ============================================================
# 8.1 温度：同一个分布，两种命运
# ============================================================

def demo_temperature():
    """
    温度只做一件事：logits 除以 T 再 softmax

    T < 1 把分布削尖（高概率的更高），T > 1 把它摊平（人人有份）。
    注意它改变的是分布的「形状」，不改变候选集合 —— 词表里的
    每一个 token 始终都有机会。
    """
    print("=" * 60)
    print("8.1 温度：同一个分布，两种命运")
    print("=" * 60)

    logits = torch.linspace(-2, 2, steps=7)

    print(f"  {'温度':<8}{'最高概率':>10}{'最低概率':>12}{'头尾差距':>12}")
    print("  " + "-" * 40)
    for t in (0.5, 1.0, 5.0):
        probas = torch.softmax(logits / t, dim=-1)
        hi, lo = probas.max().item(), probas.min().item()
        print(f"  T={t:<6}{hi:>10.4f}{lo:>12.6f}{hi / lo:>10.1f}x")

    print()
    print("  看最后一列：T=1 的时候头尾差 55 倍，T=0.5 拉开到 3000 倍，")
    print("  T=5.0 压缩到 2.2 倍 —— 温度干的就是这件事。")
    print("  但注意：长尾的概率永远不为零，只是越来越小。")
    print()


# ============================================================
# 8.2 top-p：直接把长尾砍掉
# ============================================================

def demo_top_p():
    """
    温度治不了的毛病，top-p 能治：它直接减少参赛者。

    从高到低累加概率，累计到 top_p 就划线；划线的位置之后全部清零。
    注意实现里保留的是「入场前还没越过阈值」的候选，也就是说
    刚好把累计概率顶过线的那一个也会留下。
    """
    print("=" * 60)
    print("8.2 top-p：直接把长尾砍掉")
    print("=" * 60)

    torch.manual_seed(0)
    logits = torch.randn(1, 10) * 2
    probas = torch.softmax(logits, dim=-1)

    sorted_p, order = torch.sort(probas, dim=-1, descending=True)
    print("  原始概率（降序）：")
    print("   ", "  ".join(f"{p:.4f}" for p in sorted_p[0].tolist()))
    print("  累计概率：")
    print("   ", "  ".join(f"{p:.4f}" for p in torch.cumsum(sorted_p, dim=-1)[0].tolist()))
    print()

    for p in (0.5, 0.8, 0.95, 1.0):
        kept = top_p_filter(probas.clone(), p)
        survivors = int((kept > 0).sum().item())
        print(f"  top_p={p:<5} 幸存 {survivors:2d} 个候选，最高概率 {kept.max().item():.4f}")

    print()
    print("  top_p 越小，候选越少、输出越保守；设成 1.0 等于不过滤。")
    print()


# ============================================================
# 8.3 温度 + top-p 联合生成
# ============================================================

def demo_generate(model, tokenizer, device, prompt, n_runs=3):
    """
    同一段提示词，跑几组参数配置，看输出的多样性

    规律：温度决定「票怎么分」，top-p 决定「几个人参选」。
    要多样性就两个一起放松，要稳定就一起收紧。
    """
    print("=" * 60)
    print("8.3 温度 + top-p 联合生成")
    print("=" * 60)
    print(f"  提示词：{prompt!r}")
    print()

    configs = [
        ("贪心（T=0）", 0.0, None),
        ("低温收紧（T=0.5, p=0.9）", 0.5, 0.9),
        ("高温放开（T=1.2, p=0.95）", 1.2, 0.95),
    ]

    for label, temp, top_p in configs:
        outputs = []
        for i in range(n_runs):
            torch.manual_seed(100 + i)
            idx = text_to_token_ids(prompt, tokenizer).to(device)
            out = generate_with_sampling(
                model, idx, max_new_tokens=30,
                context_size=GPT_CONFIG_124M["context_length"],
                temperature=temp, top_p=top_p,
            )
            outputs.append(token_ids_to_text(out, tokenizer)[len(prompt):])

        print(f"  【{label}】")
        if len(set(outputs)) == 1:
            print(f"    {outputs[0]!r}   ← 三次完全相同")
        else:
            for o in outputs:
                print(f"    {o!r}")
        print()

    print("  两个观察：")
    print("  1. 贪心那组三次输出一字不差 —— 没有随机性就没有多样性，")
    print("     而多样性正是下一节投票能成立的前提。")
    print("  2. 小模型经常卡进重复循环（同一串词反复吐）。这种情况光靠")
    print("     调温度救不回来，得靠多份采样 + 投票来兜底。")
    print()


# ============================================================
# 8.4 答案抽取与归一化
# ============================================================

def demo_extract():
    """
    投票之前必须先把答案「对齐格式」，否则同一个答案会被拆成好几票

    这一步看着琐碎，但它直接决定投票有没有意义：
    'x = 83' 和 '83' 是两串不同的字符，如果不归一，模型答对五次
    也可能因为写法不同而凑不出多数。
    """
    print("=" * 60)
    print("8.4 答案抽取与归一化")
    print("=" * 60)

    samples = [
        r"所以最终结果是 \boxed{83}",
        r"答案是 x = 83",
        "The answer is 83.0",
        r"我觉得是 \boxed{83.0}",
        "无法确定",
    ]

    for i, s in enumerate(samples, 1):
        raw = extract_answer(s, fallback="number")
        norm = normalize_answer(raw)
        print(f"  [{i}] 输出：{s}")
        print(f"      抽取 {raw!r} → 归一化 {norm!r}")

    print()
    print("  前四条归一化之后都是 '83'，投票时才会算作同一票。")
    print()


# ============================================================
# 8.5 自一致性投票
# ============================================================

def demo_self_consistency(model, tokenizer, device, prompt, n_samples=5):
    """
    同一道题独立答 n 遍，取得票最多的那个答案

    关键在于「平票返回 None」：五份答案各不相同的时候，说明模型
    根本没把握，这时候弃权比硬选一个更诚实。
    """
    print("=" * 60)
    print("8.5 自一致性投票")
    print("=" * 60)
    print(f"  提示词：{prompt!r}")
    print()

    result = self_consistency_vote(
        model, tokenizer, prompt, device,
        n_samples=n_samples, temperature=0.8, top_p=0.9,
        max_new_tokens=40, context_size=GPT_CONFIG_124M["context_length"],
        seed=42,
    )

    for i, ans in enumerate(result["answers"], 1):
        print(f"  [第 {i} 份] → {ans!r}")

    print()
    print(f"  票数统计：{result['counts']}")
    if result["tie"]:
        print("  平票 → final_answer = None（弃权）")
    else:
        print(f"  胜出：{result['final_answer']!r}")
    print()


# ============================================================
# 8.6 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第 8 章：推理时扩展（一）—— 采样策略与自一致性")
    print("=" * 60)
    print()

    demo_temperature()
    demo_top_p()
    demo_extract()

    device = torch.device("cpu")
    checkpoint = os.path.join(
        os.path.dirname(__file__), "..", "ch05", "gpt_model_pretrained.pth"
    )

    try:
        model, tokenizer = load_model(checkpoint, device=device)
        print()
        demo_generate(model, tokenizer, device, "The meaning of life is")
        demo_self_consistency(
            model, tokenizer, device,
            "Question: What is 12 + 7?\nAnswer:",
        )
    except FileNotFoundError as e:
        print(f"\n跳过需要模型的演示：{e}")
        print("（8.1 / 8.2 / 8.4 三节不依赖模型，已经跑完）")
