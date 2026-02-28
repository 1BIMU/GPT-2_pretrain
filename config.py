"""
GPT-2 预训练配置文件

参数来源：
- OpenAI GPT-2 论文
- nanoGPT (Karpathy)
- Hugging Face 复现
- 中文社区实践经验
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    model_size: str = "gpt2-medium"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

    # Dropout 设置
    # OpenAI原版: 0.0 (预训练时不用dropout)
    # 实验对比: 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1

    # 是否从头训练
    from_scratch: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data/openwebtext"
    train_file: str = "./data/train.bin"
    val_file: str = "./data/val.bin"
    test_file: str = "./data/test.bin"

    # 序列长度 (GPT-2 标准: 1024)
    block_size: int = 1024

    # 数据划分比例
    val_ratio: float = 0.005
    test_ratio: float = 0.005


@dataclass
class TrainingConfig:
    """
    训练配置

    参数参考:
    ┌─────────────────┬──────────────┬──────────────┬──────────────┐
    │ 参数            │ OpenAI原版   │ nanoGPT      │ 本项目默认   │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ learning_rate   │ 2.5e-4       │ 6e-4         │ 3e-4         │
    │ batch_size      │ 512          │ 480          │ 480          │
    │ warmup_steps    │ ~2000        │ 2000         │ 2000         │
    │ weight_decay    │ 0.01         │ 0.1          │ 0.1          │
    │ beta1           │ 0.9          │ 0.9          │ 0.9          │
    │ beta2           │ 0.999        │ 0.95         │ 0.95         │
    │ grad_clip       │ 1.0          │ 1.0          │ 1.0          │
    │ lr_scheduler    │ cosine       │ cosine       │ cosine       │
    └─────────────────┴──────────────┴──────────────┴──────────────┘
    """
    output_dir: str = "./output/gpt2-dropout"

    # 训练步数
    num_train_epochs: int = 1
    max_steps: int = -1  # -1表示按epoch训练，建议设为600000

    # Batch size 配置
    # 目标: 有效batch ≈ 480 sequences = 491520 tokens/batch
    # 计算: per_device_batch * gradient_accumulation * num_gpus = 480
    # 单卡示例: 12 * 40 = 480
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 12
    gradient_accumulation_steps: int = 40

    # 优化器参数 (AdamW)
    # nanoGPT 验证过的最佳设置
    learning_rate: float = 3e-4       # peak learning rate
    min_lr: float = 3e-5              # minimum lr (cosine decay终点)
    weight_decay: float = 0.1         # 比原始0.01效果更好
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95          # 比0.999更稳定
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0        # 梯度裁剪

    # 学习率调度
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000          # 线性warmup

    # 日志和保存
    logging_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 1000

    # 混合精度 (推荐bf16，如不支持则用fp16)
    fp16: bool = False
    bf16: bool = True

    # 其他
    seed: int = 42
    dataloader_num_workers: int = 4

    # 断点续训
    resume_from_checkpoint: Optional[str] = None


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()


# ============================================================================
# 预设配置 (根据不同硬件条件)
# ============================================================================

def get_config(
    model_size: str = "gpt2-medium",
    dropout: float = 0.1,
    from_scratch: bool = True,
    **kwargs
) -> Config:
    """获取配置"""
    config = Config()

    config.model.model_size = model_size
    config.model.resid_pdrop = dropout
    config.model.attn_pdrop = dropout
    config.model.embd_pdrop = dropout
    config.model.from_scratch = from_scratch

    # 更新其他参数
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)

    return config


def get_config_for_hardware(gpu_memory_gb: int, num_gpus: int = 1) -> dict:
    """
    根据硬件配置返回推荐参数

    Args:
        gpu_memory_gb: 单卡显存 (GB)
        num_gpus: GPU数量

    Returns:
        推荐的训练参数字典
    """
    # 目标有效batch size: 480 sequences
    target_batch = 480

    if gpu_memory_gb >= 80:  # A100 80GB
        micro_batch = 24
    elif gpu_memory_gb >= 40:  # A100 40GB
        micro_batch = 12
    elif gpu_memory_gb >= 24:  # RTX 3090 / 4090
        micro_batch = 8
    elif gpu_memory_gb >= 16:  # RTX 4080 / V100
        micro_batch = 4
    elif gpu_memory_gb >= 8:   # RTX 3080
        micro_batch = 2
    else:
        micro_batch = 1

    gradient_accumulation = target_batch // (micro_batch * num_gpus)

    return {
        "per_device_train_batch_size": micro_batch,
        "per_device_eval_batch_size": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": micro_batch * gradient_accumulation * num_gpus,
    }


# ============================================================================
# 模型规模参考
# ============================================================================

MODEL_SPECS = {
    "gpt2": {
        "params": "124M",
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "memory_fp16": "~8GB",
    },
    "gpt2-medium": {
        "params": "355M",
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "memory_fp16": "~16GB",
    },
    "gpt2-large": {
        "params": "774M",
        "n_layer": 36,
        "n_head": 20,
        "n_embd": 1280,
        "memory_fp16": "~32GB",
    },
    "gpt2-xl": {
        "params": "1.5B",
        "n_layer": 48,
        "n_head": 25,
        "n_embd": 1600,
        "memory_fp16": "~48GB",
    },
}


if __name__ == "__main__":
    # 打印配置信息
    print("=" * 60)
    print("GPT-2 Medium (355M) 推荐训练配置")
    print("=" * 60)

    config = Config()

    print("\n[模型配置]")
    print(f"  模型: {config.model.model_size}")
    print(f"  Dropout: {config.model.resid_pdrop}")

    print("\n[训练配置]")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Min LR: {config.training.min_lr}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Adam betas: ({config.training.adam_beta1}, {config.training.adam_beta2})")
    print(f"  Warmup steps: {config.training.warmup_steps}")
    print(f"  Gradient clip: {config.training.max_grad_norm}")

    print("\n[Batch配置]")
    print(f"  Micro batch: {config.training.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    effective = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    print(f"  Effective batch: {effective} sequences")
    print(f"  Tokens per batch: {effective * config.data.block_size:,}")

    print("\n[硬件推荐]")
    for mem in [8, 16, 24, 40, 80]:
        rec = get_config_for_hardware(mem)
        print(f"  {mem}GB GPU: batch={rec['per_device_train_batch_size']}, "
              f"accum={rec['gradient_accumulation_steps']}, "
              f"effective={rec['effective_batch_size']}")
