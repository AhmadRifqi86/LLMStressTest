#!/usr/bin/env python3
"""
fetch_questions.py — Download benchmark questions from HuggingFace datasets
and generate a bash-sourceable qcache.sh file.

The stress test sources qcache.sh to populate bash arrays; the question count
is determined at fetch time (--count N) and NOT hard-coded in any shell script.
The exponential round sizes in benchmarks.sh are computed dynamically from the
actual array lengths, so changing --count automatically adjusts all rounds.

USAGE
─────
  python3 fetch_questions.py [output_file] [--count N] [--seed S]

  output_file  Path to write qcache.sh  (default: ./qcache.sh)
  --count N    Questions per section     (default: 100)
  --seed  S    Random seed               (default: 42)

GENERATED ARRAYS
────────────────
  ARC_E_Q / ARC_E_ANS     ARC-Easy         [Clark et al. 2018]
  ARC_C_Q / ARC_C_ANS     ARC-Challenge    [Clark et al. 2018]
  HS_Q    / HS_ANS         HellaSwag        [Zellers et al. 2019]
  MMLU_Q  / MMLU_ANS       MMLU             [Hendrycks et al. 2020]
  GSM8K_Q / GSM8K_ANS      GSM8K            [Cobbe et al. 2021]
  TQA_Q   / TQA_ANS        TruthfulQA       [Lin et al. 2021]

REQUIREMENTS
────────────
  pip install datasets
"""

import sys
import argparse
import random
import re
import datetime


# ── HuggingFace datasets installation ────────────────────────────────

def _ensure_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        import subprocess
        print("Installing 'datasets' library...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


_ensure_datasets()
from datasets import load_dataset  # noqa: E402


# ── Bash string escaping ──────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape a string for use inside a bash double-quoted context."""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    s = s.replace("$", "\\$")
    s = s.replace("`", "\\`")
    return s


def _emit_arr(f, name: str, items: list[str]) -> None:
    f.write(f"{name}=(\n")
    for x in items:
        f.write(f'    "{_esc(x)}"\n')
    f.write(")\n")


def _emit_ans(f, name: str, items: list[str]) -> None:
    f.write(f"{name}=(" + " ".join(f'"{x}"' for x in items) + ")\n")


# ── Label normaliser for ARC ──────────────────────────────────────────

_LFIX = {"1": "A", "2": "B", "3": "C", "4": "D",
         "A": "A", "B": "B", "C": "C", "D": "D"}


# ── Per-dataset fetchers ──────────────────────────────────────────────

def fetch_arc(split_name: str, n: int, rng: random.Random):
    """ARC-Easy or ARC-Challenge — allenai/ai2_arc."""
    ds = [x for x in load_dataset("allenai/ai2_arc", split_name,
                                   split="test", trust_remote_code=True)
          if len(x["choices"]["label"]) == 4]
    if len(ds) < n:
        raise RuntimeError(
            f"{split_name}: only {len(ds)} 4-choice questions available (need {n})"
        )
    items = rng.sample(ds, n)
    qs, ans = [], []
    for x in items:
        labs = [_LFIX.get(l, l) for l in x["choices"]["label"]]
        opts = "\n".join(f"({l}) {t}" for l, t in zip(labs, x["choices"]["text"]))
        qs.append(x["question"] + "\n" + opts)
        ans.append(_LFIX.get(x["answerKey"], x["answerKey"]))
    return qs, ans


def fetch_hellaswag(n: int, rng: random.Random):
    """HellaSwag — Rowan/hellaswag validation split."""
    ds = list(load_dataset("Rowan/hellaswag", split="validation",
                            trust_remote_code=True))
    if len(ds) < n:
        raise RuntimeError(f"HellaSwag: only {len(ds)} examples available (need {n})")
    items = rng.sample(ds, n)
    qs, ans = [], []
    for x in items:
        opts = "\n".join(
            f"({chr(65 + i)}) {e.strip()}" for i, e in enumerate(x["endings"])
        )
        qs.append(x["ctx"].strip() + "\n" + opts)
        ans.append(chr(65 + int(x["label"])))
    return qs, ans


def fetch_mmlu(n: int, rng: random.Random):
    """MMLU — cais/mmlu all-subjects test split."""
    ds = list(load_dataset("cais/mmlu", "all", split="test",
                            trust_remote_code=True))
    if len(ds) < n:
        raise RuntimeError(f"MMLU: only {len(ds)} questions available (need {n})")
    items = rng.sample(ds, n)
    qs, ans = [], []
    for x in items:
        opts = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(x["choices"])
        )
        qs.append(x["question"] + "\n" + opts)
        ans.append(chr(65 + x["answer"]))
    return qs, ans


def fetch_gsm8k(n: int, rng: random.Random):
    """GSM8K — openai/gsm8k main test split."""
    ds = list(load_dataset("openai/gsm8k", "main", split="test",
                            trust_remote_code=True))
    if len(ds) < n:
        raise RuntimeError(f"GSM8K: only {len(ds)} questions available (need {n})")
    items = rng.sample(ds, n)
    qs, ans = [], []
    for x in items:
        m = re.search(r"####\s*([\d,]+)", x["answer"])
        qs.append(x["question"])
        ans.append(m.group(1).replace(",", "") if m else "?")
    return qs, ans


def fetch_truthfulqa(n: int, rng: random.Random):
    """TruthfulQA — truthful_qa multiple_choice validation split (MC1 format)."""
    ds = list(load_dataset("truthful_qa", "multiple_choice",
                            split="validation", trust_remote_code=True))
    rng.shuffle(ds)
    pool = []
    for x in ds:
        choices = x["mc1_targets"]["choices"]
        labels  = x["mc1_targets"]["labels"]
        if 1 not in labels:
            continue
        correct_text = choices[labels.index(1)]
        wrong = [c for c, l in zip(choices, labels) if l == 0]
        if len(wrong) < 3:
            continue
        pool.append((x["question"], correct_text, wrong))

    if len(pool) < n:
        raise RuntimeError(
            f"TruthfulQA: only {len(pool)} questions pass 4-choice filter (need {n})"
        )
    pool = pool[:n]
    qs, ans = [], []
    for question, correct_text, wrong in pool:
        four = [correct_text] + rng.sample(wrong, 3)
        rng.shuffle(four)
        opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(four))
        qs.append(question + "\n" + opts)
        ans.append(chr(65 + four.index(correct_text)))
    return qs, ans


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark questions and write qcache.sh for stress test"
    )
    parser.add_argument(
        "output", nargs="?", default="qcache.sh",
        help="Output path for the bash question cache (default: qcache.sh)",
    )
    parser.add_argument(
        "--count", type=int, default=100,
        metavar="N",
        help="Questions to fetch per benchmark section (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        metavar="S",
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    n    = args.count
    seed = args.seed
    rng  = random.Random(seed)
    out  = args.output

    sections = [
        ("ARC-Easy",     "ARC_E_Q", "ARC_E_ANS",
            lambda: fetch_arc("ARC-Easy", n, rng)),
        ("ARC-Challenge", "ARC_C_Q", "ARC_C_ANS",
            lambda: fetch_arc("ARC-Challenge", n, rng)),
        ("HellaSwag",    "HS_Q",    "HS_ANS",
            lambda: fetch_hellaswag(n, rng)),
        ("MMLU",         "MMLU_Q",  "MMLU_ANS",
            lambda: fetch_mmlu(n, rng)),
        ("GSM8K",        "GSM8K_Q", "GSM8K_ANS",
            lambda: fetch_gsm8k(n, rng)),
        ("TruthfulQA",   "TQA_Q",   "TQA_ANS",
            lambda: fetch_truthfulqa(n, rng)),
    ]

    print(f"Fetching {n} questions per section  (seed={seed})", flush=True)
    print(f"Output → {out}", flush=True)
    print()

    with open(out, "w") as f:
        f.write(
            f"# qcache.sh — auto-generated by fetch_questions.py\n"
            f"# questions_per_section={n}  seed={seed}\n"
            f"# generated={datetime.datetime.now().isoformat()}\n"
            f"# Delete this file and re-run run.sh to refresh questions.\n\n"
        )

        for section_name, q_var, a_var, fetcher in sections:
            print(f"  [{section_name}] fetching...", end=" ", flush=True)
            qs, ans = fetcher()
            assert len(qs) == n, f"{section_name}: got {len(qs)}, expected {n}"
            f.write(f"# {section_name} — {n} questions\n")
            _emit_arr(f, q_var, qs)
            _emit_ans(f, a_var, ans)
            f.write("\n")
            print(f"✓  ({len(qs)} questions)", flush=True)

    print()
    print(f"Cache written     : {out}")
    print(f"Questions/section : {n}")
    print(f"Sections          : {', '.join(s[0] for s in sections)}")
    print()
    print("Round sizes will be computed dynamically in run.sh based on these counts.")
    print("To use more questions: rm qcache.sh && ./run.sh --count 200")


if __name__ == "__main__":
    main()
