"""
数据预处理脚本
将OpenWebText数据集转换为token ids并保存为二进制文件
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from transformers import GPT2Tokenizer


def tokenize_file(args):
    """对单个文件进行tokenize"""
    file_path, tokenizer_name = args
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        if text.strip():
            tokens = tokenizer.encode(text)
            return tokens
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return []


def process_openwebtext(
    data_dir: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    num_workers: int = 8
):
    """
    处理OpenWebText数据集

    Args:
        data_dir: 解压后的OpenWebText目录
        output_dir: 输出目录
        tokenizer_name: tokenizer名称
        val_ratio: 验证集比例
        test_ratio: 测试集比例（用于计算PPL）
        num_workers: 并行处理的worker数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有txt文件
    print("Collecting files...")
    all_files = []

    # OpenWebText解压后可能有多层目录结构
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt'):
                all_files.append(os.path.join(root, fname))

    print(f"Found {len(all_files)} text files")

    if len(all_files) == 0:
        print("No .txt files found. Checking for .xz files...")
        xz_files = list(Path(data_dir).rglob("*.xz"))
        if xz_files:
            print(f"Found {len(xz_files)} .xz files. Please decompress them first:")
            print("  cd {} && find . -name '*.xz' -exec xz -d {{}} \\;".format(data_dir))
        return

    # 划分训练集、验证集和测试集
    np.random.seed(42)
    np.random.shuffle(all_files)

    test_size = int(len(all_files) * test_ratio)
    val_size = int(len(all_files) * val_ratio)

    test_files = all_files[:test_size]
    val_files = all_files[test_size:test_size + val_size]
    train_files = all_files[test_size + val_size:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # 处理函数
    def process_split(files, output_file, split_name):
        print(f"\nProcessing {split_name} split...")

        all_tokens = []
        args_list = [(f, tokenizer_name) for f in files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for tokens in tqdm(
                executor.map(tokenize_file, args_list),
                total=len(files),
                desc=f"Tokenizing {split_name}"
            ):
                if tokens:
                    all_tokens.extend(tokens)

        # 保存为二进制文件
        print(f"Total tokens in {split_name}: {len(all_tokens):,}")
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_file)
        print(f"Saved to {output_file}")

        return len(all_tokens)

    # 处理训练集、验证集和测试集
    train_tokens = process_split(
        train_files,
        os.path.join(output_dir, "train.bin"),
        "train"
    )
    val_tokens = process_split(
        val_files,
        os.path.join(output_dir, "val.bin"),
        "val"
    )
    test_tokens = process_split(
        test_files,
        os.path.join(output_dir, "test.bin"),
        "test"
    )

    # 保存元信息
    meta = {
        "tokenizer": tokenizer_name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
    }

    import json
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens: {val_tokens:,}")
    print(f"Test tokens: {test_tokens:,}")


def process_huggingface_openwebtext(
    output_dir: str,
    tokenizer_name: str = "gpt2",
    val_ratio: float = 0.005,
    test_ratio: float = 0.005,
    num_proc: int = 8
):
    """
    使用Hugging Face datasets库处理OpenWebText
    这种方式更简单，会自动下载数据
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Loading OpenWebText from Hugging Face...")
    dataset = load_dataset("openwebtext", trust_remote_code=True)

    # 加载tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return {"tokens": [tokenizer.encode(text) for text in examples["text"]]}

    print("Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # 划分训练集、验证集和测试集
    # 先分出 test，再从剩余中分出 val
    total_held_out = val_ratio + test_ratio
    split1 = tokenized["train"].train_test_split(test_size=total_held_out, seed=42)

    # 从 held_out 中分出 val 和 test
    val_test_ratio = test_ratio / total_held_out
    split2 = split1["test"].train_test_split(test_size=val_test_ratio, seed=42)

    splits = {
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"]
    }

    # 保存为二进制文件
    for split_name, split_data in splits.items():
        print(f"Processing {split_name}...")
        all_tokens = []
        for item in tqdm(split_data, desc=f"Collecting {split_name} tokens"):
            all_tokens.extend(item["tokens"])

        arr = np.array(all_tokens, dtype=np.uint16)
        output_file = os.path.join(output_dir, f"{split_name}.bin")
        arr.tofile(output_file)
        print(f"Saved {len(all_tokens):,} tokens to {output_file}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OpenWebText for GPT-2 training")
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "huggingface"],
        default="huggingface",
        help="Data source: 'local' for downloaded files, 'huggingface' for automatic download"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/openwebtext",
        help="Directory containing OpenWebText txt files (for local source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.005,
        help="Test set ratio (for PPL calculation)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    if args.source == "huggingface":
        process_huggingface_openwebtext(
            output_dir=args.output_dir,
            tokenizer_name=args.tokenizer,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            num_proc=args.num_workers
        )
    else:
        process_openwebtext(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            tokenizer_name=args.tokenizer,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            num_workers=args.num_workers
        )
