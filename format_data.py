import json
import os
import re
import time
import random
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict


def validate_output(output: str, expected_count: int) -> (bool, List[str], List[str]):
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    reasons = []

    if len(lines) != expected_count:
        reasons.append(f"Expected {expected_count} response(s), got {len(lines)}.")
    if any("?" in line for line in lines):
        reasons.append("Output contains a question mark.")
    if any(line.count(".") > 1 for line in lines):
        reasons.append(
            "One or more lines contain multiple sentences (too many periods)."
        )
    if not all(re.match(r"^[A-Z].*[.]$", line) for line in lines):
        reasons.append(
            "Each line must start with a capital letter and end with a period."
        )

    return len(reasons) == 0, reasons, lines


class TranslationCache:
    def __init__(self):
        self.cache: Dict[str, Dict[str, List[str]]] = {}

    def get_translation(self, lang: str, question: str) -> str:
        lang = lang.strip().lower()
        if lang not in self.cache:
            lang_file = f"{lang}-en/{lang}.txt"
            en_file = f"{lang}-en/en.txt"
            if not os.path.exists(lang_file) or not os.path.exists(en_file):
                raise FileNotFoundError(f"Missing bilingual files for '{lang}'")

            with (
                open(lang_file, "r", encoding="utf-8") as f1,
                open(en_file, "r", encoding="utf-8") as f2,
            ):
                self.cache[lang] = {
                    "lang_lines": [line.strip() for line in f1.readlines()],
                    "en_lines": [line.strip() for line in f2.readlines()],
                }

        lang_lines = self.cache[lang]["lang_lines"]
        en_lines = self.cache[lang]["en_lines"]

        try:
            idx = lang_lines.index(question)
        except ValueError:
            raise ValueError(
                f"Question not found in {lang}-en/{lang}.txt: {question[:60]}"
            )
        return en_lines[idx]


def generate_full_sentences(
    question: str,
    answers: List[str],
    model: str = "llama3.1",
    attempt: int = 1,
    last_error: str = None,
) -> List[str]:
    """
    Generate complete-sentence answers from a question and list of short answers.
    Enforces: one sentence only, no question marks, no line breaks.
    Allows multiple periods (for abbreviations, decimals, etc.).
    """

    grammar = """
root ::= answer_list
answer_list ::= answer (newline answer)*
answer ::= sentence
sentence ::= (!newline .)+
newline ::= "\\n"
"""

    examples = """
Examples:
Question: Where did Joe go to? Answer: the store
Response: Joe went to the store.

Question: What is the capital of France? Answer: Paris
Response: The capital of France is Paris.

Question: Who painted the Mona Lisa? Answer: Leonardo da Vinci
Response: The Mona Lisa was painted by Leonardo da Vinci.
"""

    feedback = ""
    if last_error:
        feedback = (
            f"\nPrevious attempt failed with the following issue:\n"
            f"{last_error}\n"
            f"Make sure to fix this mistake in your next response.\n"
        )

    if len(answers) == 1:
        prompt = (
            f"{examples.strip()}\n\n"
            f"{feedback}"
            f"Now write the following answer as one complete sentence. "
            f"The response must not contain a question mark (?). "
            f"The response must be one sentence only. "
            f"Any line breaks will make the answer invalid.\n\n"
            f"Question: {question}\n"
            f'Answer: "{answers[0]}"\n\n'
            f"Response:"
        )
    else:
        joined = ", ".join([f'"{a}"' for a in answers])
        prompt = (
            f"{examples.strip()}\n\n"
            f"{feedback}"
            f"Now reword the following answers so that each one is a single complete sentence. "
            f"Separate each answer with a new line. Each line must contain only one sentence and cannot contain a question mark (?). "
            f"Any line break within a single answer makes it invalid.\n\n"
            f"Question: {question}\nAnswers: {joined}\n\nResponses:"
        )

    prompt += f"\n(Note: This is attempt {attempt}. Make sure your response is clean, single-sentence, and without '?')"

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "format": "text",
            "grammar": grammar,
            "temperature": random.uniform(0.6, 1.0),
            "top_p": random.uniform(0.7, 0.95),
        },
    )

    output = response["message"]["content"].strip()

    print(f"\n LLM OUTPUT (Attempt {attempt}) ")
    print(output)

    lines = [line.strip() for line in output.split("\n") if line.strip()]

    # Must match number of answers
    if len(lines) != len(answers):
        raise ValueError(
            f"Invalid number of responses: expected {len(answers)}, got {len(lines)}\nOutput: {output}"
        )

    # Must not contain '?'
    if any("?" in line for line in lines):
        raise ValueError(f"Invalid output (contains '?'): {output}")

    # import re
    # if any(re.search(r'\.\s+[A-Z]', line) for line in lines):
    #    raise ValueError(f"Likely multiple sentences: {output}")

    return lines


def process_entry_with_retry(
    entry: dict, translator: TranslationCache, model: str, max_retries: int = 20
):
    """Process a single entry with retries, feeding LLM the reason for each failure."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            lang = entry["lang"]
            question = entry["question"]
            translated_q = translator.get_translation(lang, question)
            full_answers = generate_full_sentences(
                translated_q, entry["answers"], model, attempt, last_error
            )
            entry["translated-question"] = translated_q
            entry["full-answers"] = full_answers
            print(f"[✓] ID {entry['id']} Success: {full_answers}")
            return entry
        except Exception as e:
            last_error = str(e)
            wait_time = min(10 * attempt, 60) + random.random() * 2
            print(
                f"[Retry {attempt}/{max_retries}] ID {entry['id']} failed: {e}. Retrying in {wait_time:.1f}s...\n"
            )
            time.sleep(wait_time)

    # After exhausting retries
    raise RuntimeError(
        f"Entry {entry['id']} failed after {max_retries} retries. Last error: {last_error}"
    )


# ---------- Main Pipeline ---------- #
def process_multilang_jsonl_parallel_retry(
    input_jsonl: str,
    output_jsonl: str,
    failed_jsonl: str,
    model: str = "llama3.1",
    max_workers: int = 6,
    max_retries: int = 20,
    restart: bool = False,
):
    translator = TranslationCache()

    # Load main input data
    with open(input_jsonl, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    # --- Restart Behavior ---
    if restart:
        print(f"Ignoring previous progress and starting from scratch.")
        if os.path.exists(output_jsonl):
            os.remove(output_jsonl)
        if os.path.exists(failed_jsonl):
            os.remove(failed_jsonl)
        completed_ids = set()
        failed_entries = []
        remaining = data
    else:
        completed_ids = set()
        failed_entries = []

        # Load previously completed IDs
        if os.path.exists(output_jsonl):
            with open(output_jsonl, "r", encoding="utf-8") as fout:
                for line in fout:
                    try:
                        obj = json.loads(line)
                        completed_ids.add(obj["id"])
                    except json.JSONDecodeError:
                        continue

        # Load previously failed entries (to retry)
        if os.path.exists(failed_jsonl):
            with open(failed_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        failed_entry = json.loads(line)
                        failed_entries.append(failed_entry)
                    except json.JSONDecodeError:
                        continue

        # Filter out completed ones just in case
        failed_entries = [f for f in failed_entries if f["id"] not in completed_ids]
        remaining = [d for d in data if d["id"] not in completed_ids]

        print(
            f"Resuming from checkpoint: {len(completed_ids)} done, "
            f"{len(remaining)} new entries + {len(failed_entries)} failed entries to retry"
        )

    # Merge failed entries (retry them first)
    id_to_entry = {d["id"]: d for d in data}
    to_process = []
    for failed in failed_entries:
        if failed["id"] in id_to_entry:
            to_process.append(id_to_entry[failed["id"]])
    to_process.extend(remaining)

    if not to_process:
        print("✅ No entries left to process — all done.")
        return

    open(failed_jsonl, "w", encoding="utf-8").close()

    with (
        open(output_jsonl, "a", encoding="utf-8") as fout,
        open(failed_jsonl, "a", encoding="utf-8") as ffail,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_entry_with_retry, d, translator, model, max_retries
                ): d["id"]
                for d in to_process
            }

            for future in as_completed(futures):
                qid = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"[✓] Processed ID: {qid}")
                except Exception as e:
                    print(f"[✗] Permanent failure for ID {qid}: {e}")
                    failed_record = {"id": qid, "error": str(e)}
                    ffail.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                    ffail.flush()

    print(f"Completed all entries. Output saved to: {output_jsonl}")
    print(f"Failed entries (after {max_retries} retries) logged to: {failed_jsonl}")


# ---------- Entry Point ---------- #
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart from beginning and ignore progress",
    )
    args = parser.parse_args()

    process_multilang_jsonl_parallel_retry(
        input_jsonl="xor_train_retrieve_eng_span.jsonl",
        output_jsonl="processed_train.jsonl",
        failed_jsonl="failed_train.jsonl",
        model="gemma3n:e4b",
        max_workers=6,
        max_retries=5,
        restart=args.restart,
    )
