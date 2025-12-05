import json
import argparse
import re
import string
import random
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ollama


def normalize_answer(s):
    """Lower text and remove punctuation, articles, extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction, ground_truth):
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ground_truth.split()], prediction.split(), smoothing_function=smoothie
    )


def ask_llm(question, model, system_prompt):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""


def evaluate_jsonl(input_jsonl, model, system_prompt, sample_size=None, seed=None):
    results = []

    total_f1_short = 0
    total_bleu_short = 0
    total_f1_full = 0
    total_bleu_full = 0
    count = 0

    print(f"Reading {input_jsonl}...")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]

    total_available = len(all_lines)
    print(f"Found {total_available} entries.")

    if sample_size is not None and 0 < sample_size < total_available:
        print(f"Sampling {sample_size} entries with seed {seed}...")
        if seed is not None:
            random.seed(seed)
        lines_to_process = random.sample(all_lines, sample_size)
    else:
        lines_to_process = all_lines

    print(
        f"Starting evaluation on {len(lines_to_process)} entries using model {model}..."
    )

    for line in lines_to_process:
        entry = json.loads(line)
        qid = entry.get("id", "unknown")

        source_question = entry.get("question", "N/A")
        english_question = entry.get("translated-question", "")
        language = entry.get("language", "N/A")
        gold_short = entry.get("answers", [])
        gold_full = entry.get("full-answers", [])

        pred = ask_llm(english_question, model, system_prompt)

        best_f1_short = max([f1_score(pred, g) for g in gold_short], default=0.0)
        best_bleu_short = max([bleu_score(pred, g) for g in gold_short], default=0.0)

        best_f1_full = max([f1_score(pred, g) for g in gold_full], default=0.0)
        best_bleu_full = max([bleu_score(pred, g) for g in gold_full], default=0.0)

        total_f1_short += best_f1_short
        total_bleu_short += best_bleu_short
        total_f1_full += best_f1_full
        total_bleu_full += best_bleu_full
        count += 1

        results.append(
            {
                "id": qid,
                "language": language,
                "question": source_question,
                "translated_question": english_question,
                "pred": pred,
                "gold_short": gold_short,
                "gold_full": gold_full,
                "metrics": {
                    "f1_short": best_f1_short,
                    "bleu_short": best_bleu_short,
                    "f1_full": best_f1_full,
                    "bleu_full": best_bleu_full,
                },
            }
        )

        print(f"ID: {qid}")
        print(f"  Language:    {language}")
        print(f"  Original Q:  {source_question}")
        print(f"  English Q:   {english_question}")
        print(f"  Model Pred:  {pred}")
        print(f"  Gold Short:  {gold_short}")
        print(f"  Gold Full:   {gold_full}")
        print(f"  Short F1: {best_f1_short:.3f} | BLEU: {best_bleu_short:.3f}")
        print(f"  Full  F1: {best_f1_full:.3f} | BLEU: {best_bleu_full:.3f}")
        print("-" * 40)

    if count > 0:
        stats = {
            "total_samples": count,
            "macro_avg_f1_short": total_f1_short / count,
            "macro_avg_bleu_short": total_bleu_short / count,
            "macro_avg_f1_full": total_f1_full / count,
            "macro_avg_bleu_full": total_bleu_full / count,
        }
    else:
        stats = {"total_samples": 0, "message": "No entries processed."}

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed JSONL file")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument(
        "--system_prompt",
        default="Answer the following question in English in exactly one sentence.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_details.json",
        help="File for detailed per-question results",
    )
    parser.add_argument(
        "--stats_output",
        default="evaluation_stats.json",
        help="File for macro average statistics",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Number of random samples to evaluate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )

    args = parser.parse_args()

    results, stats = evaluate_jsonl(
        args.input, args.model, args.system_prompt, args.sample, args.seed
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing Complete.")
    print(f"Detailed results saved to: {args.output}")
    print(f"Macro statistics saved to: {args.stats_output}")

    if "macro_avg_f1_short" in stats:
        print(f"\nFinal Macro Averages:")
        print(f"  Short Answer F1:   {stats['macro_avg_f1_short']:.4f}")
        print(f"  Short Answer BLEU: {stats['macro_avg_bleu_short']:.4f}")
        print(f"  Full Answer F1:    {stats['macro_avg_f1_full']:.4f}")
        print(f"  Full Answer BLEU:  {stats['macro_avg_bleu_full']:.4f}")
