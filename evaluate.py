"""
模型评估脚本
计算困惑度和使用lm-evaluation-harness进行下游任务评估
"""
import os
import argparse
import math
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def calculate_perplexity(
    model_path: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    stride: int = 512,
    max_length: int = 1024,
    device: str = "cuda"
):
    """
    计算模型在指定数据集上的困惑度

    Args:
        model_path: 模型路径
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        split: 数据集划分
        stride: 滑动窗口步长
        max_length: 最大序列长度
        device: 设备
    """
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    print(f"Loading dataset {dataset_name}/{dataset_config}...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # 将所有文本拼接
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    print(f"Total tokens: {seq_len:,}")

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def calculate_perplexity_on_file(
    model_path: str,
    data_file: str,
    block_size: int = 1024,
    device: str = "cuda"
):
    """
    计算模型在二进制数据文件上的困惑度
    """
    import numpy as np

    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    print(f"Loading data from {data_file}...")
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    num_tokens = len(data)
    print(f"Total tokens: {num_tokens:,}")

    total_loss = 0
    total_tokens = 0
    num_batches = (num_tokens - 1) // block_size

    for i in tqdm(range(0, num_batches), desc="Calculating PPL"):
        start = i * block_size
        end = start + block_size + 1

        if end > num_tokens:
            break

        chunk = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to(device)
        input_ids = chunk[:, :-1]
        labels = chunk[:, 1:]

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        total_loss += loss.item() * block_size
        total_tokens += block_size

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def run_lm_eval(
    model_path: str,
    tasks: str = "lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge",
    num_fewshot: int = 0,
    batch_size: int = 4,
    device: str = "cuda:0"
):
    """
    使用lm-evaluation-harness进行评估

    需要先安装: pip install lm-eval
    """
    import subprocess

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks,
        "--device", device,
        "--batch_size", str(batch_size),
    ]

    if num_fewshot > 0:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def compare_models(
    model_paths: list,
    model_names: list,
    tasks: str = "lambada_openai,hellaswag,piqa,winogrande",
    device: str = "cuda"
):
    """
    对比多个模型的性能
    """
    results = {}

    for path, name in zip(model_paths, model_names):
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        # 计算WikiText-2困惑度
        ppl = calculate_perplexity(path, device=device)
        print(f"WikiText-2 PPL: {ppl:.2f}")

        results[name] = {"wikitext2_ppl": ppl}

    # 打印对比表格
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)
    print(f"{'Model':<30} {'WikiText-2 PPL':<15}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['wikitext2_ppl']:<15.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 models")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--eval_type", type=str, default="ppl",
                        choices=["ppl", "lm_eval", "both"],
                        help="Evaluation type")

    # PPL评估参数
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset for PPL evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Binary data file for PPL evaluation")

    # lm-eval参数
    parser.add_argument("--tasks", type=str,
                        default="lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge",
                        help="Tasks for lm-eval")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")

    args = parser.parse_args()

    if args.eval_type in ["ppl", "both"]:
        print("\n" + "=" * 50)
        print("Perplexity Evaluation")
        print("=" * 50)

        if args.data_file:
            ppl = calculate_perplexity_on_file(
                args.model_path,
                args.data_file,
                device=args.device
            )
            print(f"PPL on {args.data_file}: {ppl:.2f}")
        else:
            ppl = calculate_perplexity(
                args.model_path,
                args.dataset,
                args.dataset_config,
                device=args.device
            )
            print(f"PPL on {args.dataset}/{args.dataset_config}: {ppl:.2f}")

    if args.eval_type in ["lm_eval", "both"]:
        print("\n" + "=" * 50)
        print("LM Evaluation Harness")
        print("=" * 50)

        run_lm_eval(
            args.model_path,
            args.tasks,
            args.num_fewshot,
            args.batch_size,
            args.device
        )


if __name__ == "__main__":
    main()
