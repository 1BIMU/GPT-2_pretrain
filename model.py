"""
模型定义和工具函数
"""
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer


def create_model(
    model_size: str = "gpt2-medium",
    resid_pdrop: float = 0.1,
    attn_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    from_scratch: bool = True
):
    """
    创建GPT-2模型

    Args:
        model_size: 模型大小 (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        resid_pdrop: 残差连接dropout
        attn_pdrop: 注意力dropout
        embd_pdrop: embedding dropout
        from_scratch: 是否从头训练（随机初始化）

    Returns:
        model: GPT2LMHeadModel
        config: GPT2Config
    """
    # 加载配置
    config = GPT2Config.from_pretrained(model_size)

    # 设置dropout
    config.resid_pdrop = resid_pdrop
    config.attn_pdrop = attn_pdrop
    config.embd_pdrop = embd_pdrop

    print(f"Model config:")
    print(f"  - Model size: {model_size}")
    print(f"  - Hidden size: {config.n_embd}")
    print(f"  - Num layers: {config.n_layer}")
    print(f"  - Num heads: {config.n_head}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Dropout: resid={resid_pdrop}, attn={attn_pdrop}, embd={embd_pdrop}")

    if from_scratch:
        # 随机初始化
        print("Initializing model from scratch...")
        model = GPT2LMHeadModel(config)
    else:
        # 加载预训练权重
        print(f"Loading pretrained weights from {model_size}...")
        model = GPT2LMHeadModel.from_pretrained(model_size, config=config)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {num_trainable:,}")

    return model, config


def get_tokenizer(model_size: str = "gpt2"):
    """获取tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_size)
    # 设置pad token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# 模型大小对应的参数量
MODEL_SIZES = {
    "gpt2": "124M",
    "gpt2-medium": "355M",
    "gpt2-large": "774M",
    "gpt2-xl": "1.5B"
}
