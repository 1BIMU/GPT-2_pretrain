# GPT-2 Medium (355M) 训练参数汇总

## 参数来源对比

| 参数 | OpenAI 原版 | nanoGPT (Karpathy) | 本项目配置 | 说明 |
|------|------------|-------------------|-----------|------|
| **学习率** | 2.5e-4 | 6e-4 (peak) | 3e-4 | nanoGPT用更高lr配合更大weight_decay |
| **最小学习率** | - | 6e-5 | 3e-5 | cosine decay终点，通常为peak的10% |
| **Batch Size** | 512 | 480 | 480 | 有效batch (sequences) |
| **Tokens/Batch** | ~524K | ~491K | ~491K | batch_size × 1024 |
| **Warmup Steps** | ~2000 | 2000 | 2000 | 线性warmup |
| **Weight Decay** | 0.01 | 0.1 | 0.1 | 0.1比0.01效果更好 |
| **Adam β1** | 0.9 | 0.9 | 0.9 | 标准值 |
| **Adam β2** | 0.999 | 0.95 | 0.95 | 0.95更稳定，现代LLM标准 |
| **梯度裁剪** | 1.0 | 1.0 | 1.0 | 标准值 |
| **LR Schedule** | cosine | cosine | cosine | 余弦退火 |
| **训练步数** | ~800K | ~600K | 按数据量 | OpenWebText约600K步 |
| **Dropout** | 0.0 | 0.0 | 0.0/0.1 | 预训练通常不用，本实验对比 |
| **序列长度** | 1024 | 1024 | 1024 | GPT-2标准 |

---

## 模型架构 (GPT-2 Medium)

| 参数 | 值 |
|------|-----|
| 参数量 | 355M |
| 层数 (n_layer) | 24 |
| 隐藏维度 (n_embd) | 1024 |
| 注意力头数 (n_head) | 16 |
| 词表大小 | 50257 |
| 序列长度 | 1024 |

---

## 本项目最终配置

```python
# config.py 中的默认配置

# 模型
model_size = "gpt2-medium"
dropout = 0.1  # 实验组，对照组为0.0

# 优化器 (AdamW)
learning_rate = 3e-4      # peak lr
min_lr = 3e-5             # cosine decay终点
weight_decay = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.95         # 比0.999更稳定
adam_epsilon = 1e-8
max_grad_norm = 1.0       # 梯度裁剪

# Batch配置
per_device_train_batch_size = 12
gradient_accumulation_steps = 40
# 有效batch = 12 × 40 = 480 sequences = 491,520 tokens

# 学习率调度
lr_scheduler_type = "cosine"
warmup_steps = 2000

# 混合精度
bf16 = True  # 优先使用bf16，不支持则用fp16
```

---

## 不同硬件的推荐配置

| GPU显存 | micro_batch | gradient_accum | 有效batch | 说明 |
|---------|-------------|----------------|-----------|------|
| 8GB (RTX 3080) | 2 | 240 | 480 | 需要较多累积步 |
| 16GB (V100/4080) | 4 | 120 | 480 | |
| 24GB (3090/4090) | 8 | 60 | 480 | |
| 40GB (A100-40G) | 12 | 40 | 480 | 推荐配置 |
| 80GB (A100-80G) | 24 | 20 | 480 | |

---

## 关键发现

1. **β2=0.95** 是现代LLM训练的重要改进，比原始的0.999更稳定
2. **Weight decay=0.1** 配合更高学习率效果更好
3. **Batch size** 对大模型训练很关键，建议 tokens/batch ≥ 0.5M
4. **Dropout** OpenAI预训练时设为0.0，本实验对比0.1的效果
5. **bf16** 比fp16更稳定，推荐在支持的硬件上使用

---

## 参考来源

- OpenAI GPT-2 论文: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- nanoGPT: https://github.com/karpathy/nanoGPT
- Hugging Face Transformers: https://huggingface.co/gpt2-medium
- 中文社区实践 (知乎/CSDN)
