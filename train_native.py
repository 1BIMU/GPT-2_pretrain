"""
GPT-2 预训练脚本 (PyTorch原生版本)
不依赖Hugging Face Trainer，更灵活可控
"""
import os
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from model import create_model, get_tokenizer
from dataset import PretrainDataset, StreamingDataset, DataCollatorForLM


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    学习率调度：线性warmup + cosine decay
    """
    # Warmup阶段
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Decay阶段
    if step > max_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device, ctx):
    """评估函数"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with ctx:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()
    return avg_loss, perplexity


def train(config):
    """训练主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)

    # 创建模型
    model, model_config = create_model(
        model_size=config.model.model_size,
        resid_pdrop=config.model.resid_pdrop,
        attn_pdrop=config.model.attn_pdrop,
        embd_pdrop=config.model.embd_pdrop,
        from_scratch=config.model.from_scratch
    )
    model = model.to(device)

    # 获取tokenizer
    tokenizer = get_tokenizer(config.model.model_size)

    # 创建数据集
    print("\nLoading datasets...")
    train_dataset = PretrainDataset(
        config.data.train_file,
        config.data.block_size
    )
    val_dataset = PretrainDataset(
        config.data.val_file,
        config.data.block_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=True
    )

    # 计算训练步数
    steps_per_epoch = len(train_loader) // config.training.gradient_accumulation_steps
    if config.training.max_steps > 0:
        max_steps = config.training.max_steps
    else:
        max_steps = steps_per_epoch * config.training.num_train_epochs

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Max steps: {max_steps}")

    # 优化器 (使用nanoGPT验证过的参数)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon
    )

    # 混合精度
    use_amp = config.training.fp16 or config.training.bf16
    if config.training.bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif config.training.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    scaler = GradScaler(enabled=config.training.fp16)  # bf16不需要scaler
    ctx = autocast(dtype=dtype) if use_amp else nullcontext()

    # TensorBoard
    writer = SummaryWriter(os.path.join(config.training.output_dir, "logs"))

    # 训练循环
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(config.training.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_train_epochs}")

        for micro_step, batch in enumerate(train_loader):
            # 获取学习率 (使用min_lr配置)
            lr = get_lr(
                global_step,
                config.training.warmup_steps,
                max_steps,
                config.training.learning_rate,
                config.training.min_lr
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # 前向传播
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / config.training.gradient_accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积
            if (micro_step + 1) % config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm
                )

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                # 日志
                if global_step % config.training.logging_steps == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        global_step *
                        config.training.per_device_train_batch_size *
                        config.training.gradient_accumulation_steps *
                        config.data.block_size
                    ) / elapsed

                    print(
                        f"Step {global_step}/{max_steps} | "
                        f"Loss: {loss.item() * config.training.gradient_accumulation_steps:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f}"
                    )

                    writer.add_scalar("train/loss", loss.item() * config.training.gradient_accumulation_steps, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step)

                # 评估
                if global_step % config.training.eval_steps == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device, ctx)
                    print(f"  Eval - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")

                    writer.add_scalar("eval/loss", val_loss, global_step)
                    writer.add_scalar("eval/perplexity", val_ppl, global_step)

                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(config.training.output_dir, "best")
                        os.makedirs(save_path, exist_ok=True)
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"  Saved best model to {save_path}")

                # 定期保存
                if global_step % config.training.save_steps == 0:
                    save_path = os.path.join(
                        config.training.output_dir,
                        f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

                    # 保存优化器状态
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                    }, os.path.join(save_path, "training_state.pt"))

                    print(f"  Saved checkpoint to {save_path}")

                # 检查是否达到最大步数
                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    # 保存最终模型
    print("\nSaving final model...")
    save_path = os.path.join(config.training.output_dir, "final")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 最终评估
    val_loss, val_ppl = evaluate(model, val_loader, device, ctx)
    print(f"\nFinal Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val PPL: {val_ppl:.2f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")

    writer.close()
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 (PyTorch native)")

    # 模型参数
    parser.add_argument("--model_size", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--from_scratch", action="store_true", default=True)

    # 数据参数
    parser.add_argument("--train_file", type=str, default="./data/train.bin")
    parser.add_argument("--val_file", type=str, default="./data/val.bin")
    parser.add_argument("--block_size", type=int, default=1024)

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/gpt2-dropout")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    parser.add_argument("--fp16", action="store_true", default=True)

    args = parser.parse_args()

    # 创建配置
    config = get_config(
        model_size=args.model_size,
        dropout=args.dropout,
        from_scratch=args.from_scratch,
    )

    # 更新配置
    config.data.train_file = args.train_file
    config.data.val_file = args.val_file
    config.data.block_size = args.block_size

    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.num_epochs
    config.training.max_steps = args.max_steps
    config.training.per_device_train_batch_size = args.batch_size
    config.training.per_device_eval_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.learning_rate = args.learning_rate
    config.training.warmup_steps = args.warmup_steps
    config.training.weight_decay = args.weight_decay
    config.training.logging_steps = args.logging_steps
    config.training.save_steps = args.save_steps
    config.training.eval_steps = args.eval_steps
    config.training.fp16 = args.fp16

    # 打印配置
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Model: {config.model.model_size}")
    print(f"Dropout: {config.model.resid_pdrop}")
    print(f"From scratch: {config.model.from_scratch}")
    print(f"Block size: {config.data.block_size}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"FP16: {config.training.fp16}")
    print(f"Output dir: {config.training.output_dir}")
    print("=" * 50)

    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
