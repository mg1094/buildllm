"""
公共模块：GPT 模型组件、生成函数和训练工具
"""

from .gpt_model import (
    LayerNorm,
    GELU,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    GPTModel,
)

from .generation import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    generate,
)

from .training import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    train_model_simple,
)
