# 2.2 文本分词 + 2.3 Token IDs + 2.4 特殊上下文Token
import re
import torch


# ============================================================
# 2.2 文本分词
# ============================================================
print("=" * 50)
print("2.2 文本分词")
print("=" * 50)

# 简单示例
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(f"\n原文: {text}")
print(f"分词结果: {result}")


# ============================================================
# 2.3 将 tokens 转换为 token IDs
# ============================================================
print("\n" + "=" * 50)
print("2.3 构建词汇表并转换为 token IDs")
print("=" * 50)

# 使用课程示例文本
raw_text = """I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so
it was no great surprise to me to hear that, in"""

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(f"\n分词数量: {len(preprocessed)}")
print(f"前30个token: {preprocessed[:30]}")

# 构建词汇表
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"\n词汇表大小: {vocab_size}")

vocab = {token: idx for idx, token in enumerate(all_words)}
print(f"\n词汇表前20项:")
for i, (token, idx) in enumerate(vocab.items()):
    if i >= 20:
        break
    print(f"  '{token}': {idx}")


# ============================================================
# SimpleTokenizer V1 (基础版)
# ============================================================
print("\n" + "=" * 50)
print("2.3 SimpleTokenizer V1")
print("=" * 50)


class SimpleTokenizerV1:
    """
    基础分词器 V1
    - 使用正则表达式分词
    - 词汇表仅包含训练文本中出现的token
    - 遇到未知token会报错
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 测试 V1 (只使用词汇表中的token)
tokenizer_v1 = SimpleTokenizerV1(vocab)

text1 = "Jack Gisburn rather a cheap genius"
ids = tokenizer_v1.encode(text1)
print(f"\n编码测试: '{text1}'")
print(f"Token IDs: {ids}")

decoded = tokenizer_v1.decode(ids)
print(f"解码结果: '{decoded}'")

# V1 的缺陷：遇到未知token会报错
text2 = "Hello, do you like tea?"
print(f"\n测试未知token: '{text2}'")
try:
    tokenizer_v1.encode(text2)
except KeyError as e:
    print(f"  ❌ 报错: KeyError {e} (词汇表中不存在的token)")


# ============================================================
# SimpleTokenizer V2 (支持未知token和EOS)
# ============================================================
print("\n" + "=" * 50)
print("2.4 SimpleTokenizer V2 (支持特殊token)")
print("=" * 50)


class SimpleTokenizerV2:
    """
    改进版分词器 V2
    - 添加 <|unk|> 处理未知token
    - 添加 <|endoftext|> 标记文本结束
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # 未知token替换为 <|unk|>
        preprocessed = [
            token if token in self.str_to_int else "<|unk|>"
            for token in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 扩展词汇表，添加特殊token
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_v2 = {token: idx for idx, token in enumerate(all_tokens)}

print(f"\n扩展后词汇表大小: {len(vocab_v2)}")
print(f"特殊token: '<|endoftext|>': {vocab_v2['<|endoftext|>']}, '<|unk|>': {vocab_v2['<|unk|>']}")

tokenizer_v2 = SimpleTokenizerV2(vocab_v2)

# 测试未知token
text3 = "Hello, do you like tea?"
ids3 = tokenizer_v2.encode(text3)
print(f"\n编码测试 (含未知token): '{text3}'")
print(f"Token IDs: {ids3}")
print(f"解码结果: '{tokenizer_v2.decode(ids3)}'")

# 测试 <|endoftext|>
text4 = "Every effort moves you<|endoftext|>Every day holds a"
# 手动添加 endoftext token
ids4 = tokenizer_v2.encode("Every effort moves you") + [vocab_v2["<|endoftext|>"]] + tokenizer_v2.encode("Every day holds a")
print(f"\n编码测试 (含endoftext): '{text4}'")
print(f"Token IDs: {ids4}")


# ============================================================
# 2.5 字节对编码 (BPE) - 使用 tiktoken
# ============================================================
print("\n" + "=" * 50)
print("2.5 字节对编码 (BPE) - tiktoken")
print("=" * 50)

try:
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    print("\n✅ tiktoken 加载成功")

    # 测试 BPE 分词
    text_bpe = "Hello, do you like tea? A pineapple costs $1.50."
    int_encoding = tokenizer.encode(text_bpe, allowed_special={"<|endoftext|>"})
    print(f"\n原文: {text_bpe}")
    print(f"Token IDs: {int_encoding}")
    print(f"Token 数量: {len(int_encoding)}")

    # 解码
    decoded_bpe = tokenizer.decode(int_encoding)
    print(f"解码结果: {decoded_bpe}")

    # 获取 token 字符串
    tokens = [tokenizer.decode([i]) for i in int_encoding]
    print(f"Token 列表: {tokens}")

except ImportError:
    print("\n⚠️ tiktoken 未安装，运行: pip install tiktoken")
