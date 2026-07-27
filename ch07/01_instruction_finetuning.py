# 第7章：指令遵循微调
# 基于预训练 GPT 模型进行指令微调

import json
import os
import re
import time
import sys
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 确保输出立即刷新
sys.stdout.reconfigure(line_buffering=True)

# 添加项目根目录到路径，以便导入 common 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.gpt_model import GPTModel
from common.generation import text_to_token_ids, token_ids_to_text, generate
from common.training import calc_loss_batch, calc_loss_loader, train_model_simple


# ============================================================
# 7.2 为监督指令微调准备数据集
# ============================================================

def download_and_load_file(file_path, url):
    """下载并加载 JSON 格式的数据集"""
    if not os.path.exists(file_path):
        print(f"正在下载数据集...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"数据集已保存至: {file_path}")
    else:
        print(f"{file_path} 已存在，跳过下载。")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data


def format_input(entry):
    """
    将数据条目格式化为 Alpaca 风格的输入文本
    
    格式:
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    {instruction}
    
    ### Input:
    {input}
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    
    return instruction_text + input_text


# ============================================================
# 7.3 将数据组织成训练批次
# ============================================================

class InstructionDataset(Dataset):
    """指令数据集"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        # 预分词所有文本
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    自定义的 collate 函数，用于处理指令数据集的批次
    
    关键特性:
    1. 将批次中的样本填充到相同长度
    2. 生成输入和目标序列（目标序列右移一位）
    3. 将目标序列中的填充 token 替换为 ignore_index（计算损失时忽略）
    4. 可选的最大长度截断
    """
    # 找到批次中最长的序列长度
    batch_max_length = max(len(item) + 1 for item in batch)
    
    # 准备输入和目标列表
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        # 添加一个 <|endoftext|> token
        new_item += [pad_token_id]
        
        # 填充到最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # 输入：截断最后一个 token
        inputs = torch.tensor(padded[:-1])
        # 目标：右移一位
        targets = torch.tensor(padded[1:])
        
        # 将目标序列中的填充 token（除了第一个）替换为 ignore_index
        # 这样在计算交叉熵损失时会忽略这些位置
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        # 可选的最大长度截断
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    # 转换为张量并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor


# ============================================================
# 7.7 提取并保存响应
# ============================================================

def generate_and_save_responses(model, test_data, tokenizer, device, output_file="test_responses.json"):
    """
    在测试集上生成响应并保存
    
    Args:
        model: GPT 模型
        test_data: 测试数据集
        tokenizer: 分词器
        device: 设备
        output_file: 输出文件路径
    
    Returns:
        带有模型响应的测试数据
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=context_size,
            eos_id=50256
        )
        
        generated_text = token_ids_to_text(token_ids, tokenizer)
        
        # 提取模型生成的响应部分
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        
        test_data[i]["model_response"] = response_text
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 模型响应已保存至: {output_file}")
    model.train()
    
    return test_data


# ============================================================
# 7.8 评估指令微调后的 LLM
# ============================================================

def check_response_quality(entry, verbose=True):
    """
    检查模型响应质量（简单示例）
    
    Args:
        entry: 包含 instruction, input, output 和 model_response 的字典
        verbose: 是否打印详细信息
    
    Returns:
        质量评分（简单匹配）
    """
    instruction = entry["instruction"]
    input_text = entry["input"]
    expected = entry["output"]
    generated = entry.get("model_response", "")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"指令: {instruction}")
        if input_text:
            print(f"输入: {input_text}")
        print(f"期望输出: {expected}")
        print(f"模型生成: {generated}")
        print(f"{'='*60}")
    
    # 简单质量检查：生成的文本是否与期望输出有一定的重叠
    # 实际应用中应该使用更复杂的评估方法
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())
    
    if len(expected_words) == 0:
        return 0.0
    
    overlap = len(expected_words.intersection(generated_words))
    score = overlap / len(expected_words)
    
    return score


def evaluate_model_on_test_set(model, test_data, tokenizer, device, num_samples=5):
    """
    在测试集上评估模型
    
    Args:
        model: GPT 模型
        test_data: 测试数据集
        tokenizer: 分词器
        device: 设备
        num_samples: 评估的样本数
    """
    print(f"\n{'='*60}")
    print(f"在 {num_samples} 个测试样本上评估模型")
    print(f"{'='*60}")
    
    total_score = 0.0
    
    for i in range(min(num_samples, len(test_data))):
        entry = test_data[i]
        input_text = format_input(entry)
        context_size = model.pos_emb.weight.shape[0]
        
        # 生成响应
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=context_size,
            eos_id=50256
        )
        
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        
        # 保存响应
        test_data[i]["model_response"] = response_text
        
        # 评估质量
        score = check_response_quality(test_data[i], verbose=True)
        total_score += score
        print(f"简单匹配得分: {score:.3f}\n")
    
    avg_score = total_score / min(num_samples, len(test_data))
    print(f"\n平均匹配得分: {avg_score:.3f}")
    
    return test_data


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第7章：指令遵循微调")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # 7.2 准备数据集
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.2 准备数据集")
    print("=" * 60)
    
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    
    file_path = "instruction-data.json"
    data = download_and_load_file(file_path, url)
    
    print(f"\n数据集条目数: {len(data)}")
    print(f"\n示例条目:\n{data[50]}")
    print(f"\n另一个示例（input 为空）:\n{data[999]}")
    
    # 测试 format_input 函数
    print("\n" + "-" * 40)
    print("测试 Alpaca 风格提示格式")
    print("-" * 40)
    
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(f"\n格式化后的输入+响应:\n{model_input}{desired_response}")
    
    # 划分数据集
    print("\n" + "-" * 40)
    print("划分数据集")
    print("-" * 40)
    
    train_portion = int(len(data) * 0.85)  # 85% 用于训练
    test_portion = int(len(data) * 0.1)    # 10% 用于测试
    val_portion = len(data) - train_portion - test_portion  # 剩余 5% 用于验证
    
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 保存数据集
    with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open("val_data.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    
    # ----------------------------------------------------------
    # 7.3-7.4 创建数据加载器
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.3-7.4 创建数据加载器")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 创建数据集
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    
    print(f"\n训练集样本 0 的 token 数: {len(train_dataset[0])}")
    
    # 创建数据加载器
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 使用 functools.partial 创建带有默认参数的 collate 函数
    from functools import partial
    
    custom_collate_fn_with_defaults = partial(
        custom_collate_fn,
        pad_token_id=50256,
        device=device,
        allowed_max_length=1024  # 限制最大长度以节省内存
    )
    
    torch.manual_seed(123)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_with_defaults,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_with_defaults,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_with_defaults,
        drop_last=False
    )
    
    print(f"\n训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 测试批次维度
    for inputs, targets in train_loader:
        print(f"\n输入批次维度: {inputs.shape}")
        print(f"目标批次维度: {targets.shape}")
        print(f"输入样本: {inputs[0][:20]}...")
        print(f"目标样本: {targets[0][:20]}...")
        break
    
    # ----------------------------------------------------------
    # 7.5 加载预训练的 LLM
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.5 加载预训练的 LLM")
    print("=" * 60)
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    model = GPTModel(GPT_CONFIG_124M)
    
    # 尝试加载预训练权重
    pretrained_path = "../ch05/gpt_model_pretrained.pth"
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("未找到预训练权重，使用随机初始化")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # ----------------------------------------------------------
    # 7.6 指令微调 LLM
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.6 开始指令微调")
    print("=" * 60)
    
    # 测试生成（微调前）
    print("\n" + "-" * 40)
    print("微调前模型生成测试")
    print("-" * 40)
    
    start_context = format_input({
        "instruction": "What is an antonym of 'difficult'?",
        "input": ""
    })
    
    context_size = model.pos_emb.weight.shape[0]
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=50,
        context_size=context_size,
        temperature=1.0,
        top_k=50,
        eos_id=50256
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\n输入:\n{start_context}")
    print(f"\n生成:\n{generated_text}")
    
    # 开始训练
    print("\n" + "-" * 40)
    print("开始训练")
    print("-" * 40)
    
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    
    num_epochs = 2
    eval_freq = 1
    eval_iter = 3
    
    start_time = time.time()
    tokens_seen, train_losses, val_losses = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer
    )
    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed:.1f} 秒")
    
    # ----------------------------------------------------------
    # 7.7 提取并保存响应
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.7 在测试集上生成响应")
    print("=" * 60)
    
    # 只在前几个测试样本上生成响应（节省时间）
    test_subset = test_data[:3].copy()
    test_with_responses = generate_and_save_responses(
        model, test_subset, tokenizer, device,
        output_file="test_responses.json"
    )
    
    # ----------------------------------------------------------
    # 7.8 评估指令微调后的 LLM
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("7.8 评估模型")
    print("=" * 60)
    
    evaluate_model_on_test_set(model, test_data[:3], tokenizer, device, num_samples=3)
    
    # 保存模型
    print("\n" + "=" * 60)
    print("保存指令微调后的模型")
    print("=" * 60)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "gpt_instruction_finetuned.pth")
    print("模型已保存至: gpt_instruction_finetuned.pth")
    
    print("\n" + "=" * 60)
    print("第7章完成!")
    print("=" * 60)
