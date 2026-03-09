"""
创建小型测试数据集，用于验证 AGD 训练流程是否正确

生成 train.bin 和 val.bin，各包含少量随机 token
不需要网络连接
"""
import numpy as np
import os


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    # 尝试使用 tokenizer（离线缓存），如果失败则使用随机 token
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

        # 5 条简短的测试文本
        texts = [
            "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing purposes.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Adversarial dropout is a regularization technique based on a min-max optimization framework.",
            "The generator learns to produce dropout masks that maximize the task loss under sparsity constraints.",
            "Deep neural networks have achieved remarkable success in computer vision and natural language processing.",
        ]

        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        if len(all_tokens) == 0:
            raise ValueError("Tokenizer returned empty tokens")
        print(f"使用 GPT-2 tokenizer 编码，总 token 数: {len(all_tokens)}")

    except Exception as e:
        print(f"⚠️ 无法加载 tokenizer ({e})，使用随机 token 代替")
        # GPT-2 vocab_size = 50257，生成合理范围的随机 token
        all_tokens = list(np.random.randint(0, 50257, size=200))
        print(f"随机生成 token 数: {len(all_tokens)}")

    # 重复若干次，确保至少能组成几个 block (block_size=128 用于测试)
    min_tokens = 2 * (128 + 1) * 5  # 2 batch, 5 samples 留余量
    repeats = max(1, min_tokens // len(all_tokens) + 1)
    all_tokens = all_tokens * repeats
    print(f"重复 {repeats} 次后 token 数: {len(all_tokens)}")

    tokens_array = np.array(all_tokens, dtype=np.uint16)

    # 80% train, 20% val
    split = int(len(tokens_array) * 0.8)
    train_tokens = tokens_array[:split]
    val_tokens = tokens_array[split:]

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"✅ 训练集: {len(train_tokens)} tokens → {train_path}")
    print(f"✅ 验证集: {len(val_tokens)} tokens → {val_path}")
    print(f"✅ 测试数据已创建完毕！")


if __name__ == "__main__":
    main()
