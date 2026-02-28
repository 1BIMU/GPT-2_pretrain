"""
数据集类
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    """
    预训练数据集
    从二进制文件中读取token ids
    """

    def __init__(self, data_file: str, block_size: int = 1024):
        """
        Args:
            data_file: 二进制数据文件路径 (.bin)
            block_size: 序列长度
        """
        self.block_size = block_size

        # 使用内存映射读取大文件
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)

        # 计算样本数量
        self.num_samples = (self.num_tokens - 1) // block_size

        print(f"Loaded {self.num_tokens:,} tokens from {data_file}")
        print(f"Number of samples: {self.num_samples:,}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取连续的block_size+1个token
        start = idx * self.block_size
        end = start + self.block_size + 1

        # 转换为tensor
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        # input和target错开一位
        x = chunk[:-1]
        y = chunk[1:]

        return {"input_ids": x, "labels": y}


class StreamingDataset(Dataset):
    """
    流式数据集，随机采样
    适合大规模数据训练
    """

    def __init__(
        self,
        data_file: str,
        block_size: int = 1024,
        num_samples: int = None
    ):
        self.block_size = block_size
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)

        # 如果指定了样本数量，使用指定值；否则根据数据量计算
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = (self.num_tokens - 1) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机选择起始位置
        max_start = self.num_tokens - self.block_size - 1
        start = np.random.randint(0, max_start)
        end = start + self.block_size + 1

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        x = chunk[:-1]
        y = chunk[1:]

        return {"input_ids": x, "labels": y}


def get_dataloaders(
    train_file: str,
    val_file: str,
    block_size: int = 1024,
    train_batch_size: int = 4,
    val_batch_size: int = 4,
    num_workers: int = 4,
    streaming: bool = False,
    train_samples: int = None
):
    """
    获取数据加载器

    Args:
        train_file: 训练数据文件
        val_file: 验证数据文件
        block_size: 序列长度
        train_batch_size: 训练batch size
        val_batch_size: 验证batch size
        num_workers: DataLoader worker数量
        streaming: 是否使用流式数据集
        train_samples: 训练样本数量（仅streaming模式有效）
    """
    if streaming:
        train_dataset = StreamingDataset(train_file, block_size, train_samples)
    else:
        train_dataset = PretrainDataset(train_file, block_size)

    val_dataset = PretrainDataset(val_file, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


class DataCollatorForLM:
    """
    用于Hugging Face Trainer的数据整理器
    """

    def __call__(self, examples):
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        labels = torch.stack([ex["labels"] for ex in examples])

        return {
            "input_ids": input_ids,
            "labels": labels
        }
