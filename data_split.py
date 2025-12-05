import json
import argparse
import random


def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def split_jsonl(input_path, seed=42, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must add to 1"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(data)

    N = len(data)
    n_train = int(N * train_ratio)
    n_dev = int(N * dev_ratio)

    train_set = data[:n_train]
    dev_set = data[n_train : n_train + n_dev]
    test_set = data[n_train + n_dev :]

    save_jsonl("train.jsonl", train_set)
    save_jsonl("dev.jsonl", dev_set)
    save_jsonl("test.jsonl", test_set)

    print(f"Total entries: {N}")
    print(f"Train: {len(train_set)}  Dev: {len(dev_set)}  Test: {len(test_set)}")
    print("Saved as train.jsonl, dev.jsonl, test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="The processed JSONL file to split"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    args = parser.parse_args()

    split_jsonl(
        args.input, args.seed, args.train_ratio, args.dev_ratio, args.test_ratio
    )
