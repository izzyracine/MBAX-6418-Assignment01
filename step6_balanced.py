"""MBAX 6418 - Assignment 1, Step 6: three-class sentiment on a balanced,
reproducible sample of the full Amazon 2023 Gift Cards dataset.

Changes vs earlier steps
------------------------
* 3 classes: POSITIVE (rating 4-5), NEUTRAL (rating 3), NEGATIVE (rating 1-2).
* The LLM prompt returns POSITIVE / NEUTRAL / NEGATIVE (rating never sent).
* Instead of the first 100 dataset lines, a balanced sample of ~50 reviews
  per class is drawn from the FULL dataset with a fixed random seed, so the
  exact same reviews are selected on every run.

Outputs (Step 2 and Step 5 files are never touched)
---------------------------------------------------
  step6_sample.csv    the balanced sample: dataset_index, asin, title, text,
                      rating, class
  step6_results.csv   sample plus 3-class LLM predictions: ...,
                      true_class, predicted_class, correct, llm_raw

Usage:
    python step6_balanced.py [--k 50] [--seed 42]
"""

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

from review_prompt import build_3class_prompt, load_env_file, parse_3class_label

DATA_PATH = "data/Gift_Cards.jsonl.gz"
SAMPLE_PATH = "step6_sample.csv"
RESULTS_PATH = "step6_results.csv"
REQUEST_DELAY_S = 0.2  # be polite to the shared class endpoint

# Full-dataset class counts from the Step 2 download (152,410 reviews).
# Used only for reference in the printed comparison.
FULL_DATASET_TOTAL = 152_410

load_env_file()
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

CLASSES = ("POSITIVE", "NEUTRAL", "NEGATIVE")

# Step 2 binary run (imbalanced, first 100 dataset lines) -- from the saved
# step2_results.csv; kept here so the comparison is printed alongside.
BINARY_REF = {
    "n": 100,
    "mix": "93 POSITIVE : 7 NEGATIVE",
    "overall": "98.0%",
    "per_class": {"POSITIVE": "92/93 (98.9%)", "NEGATIVE": "6/7 (85.7%)"},
}


def class_of(rating) -> str:
    """Assignment rule: 4-5 POSITIVE, 3 NEUTRAL, 1-2 NEGATIVE."""
    r = float(rating)
    if r >= 4:
        return "POSITIVE"
    if r == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def build_sample(k: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    """Draw a deterministic, balanced sample from the FULL dataset.

    Pass 1: bucket every dataset line index by its class and print the
    full-dataset class distribution. Pass 2: read again and keep only the
    seed-picked lines, in dataset order. Returns (rows, dist_full).
    """
    buckets = {c: [] for c in CLASSES}
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as fh:
        for ln, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            rating = rec.get("rating")
            if rating is None:
                continue
            buckets[class_of(rating)].append(ln)

    dist_full = {c: len(buckets[c]) for c in CLASSES}
    dist_full["total"] = sum(dist_full.values())

    rng = random.Random(seed)
    picked = {}
    for c in CLASSES:
        n_take = min(k, len(buckets[c]))
        picked[c] = set(rng.sample(buckets[c], n_take))
    keep = set().union(*picked.values())

    rows = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as fh:
        for ln, line in enumerate(fh):
            if ln not in keep:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            rows.append({
                "dataset_index": ln,
                "asin": rec.get("asin", ""),
                "title": (rec.get("title") or "").strip(),
                "text": (rec.get("text") or "").strip(),
                "rating": float(rec.get("rating")),
                "class": class_of(rec.get("rating")),
            })
    rows.sort(key=lambda r: r["dataset_index"])
    return rows, dist_full


def chat(messages: list[dict], retries: int = 2) -> str:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt < retries:
                print(f"    (retry {attempt + 1} after: {e})", flush=True)
                time.sleep(3 * (attempt + 1))
    raise last_err


def report(results: list[dict], dist_full: dict[str, int],
           sample_counts: dict[str, int], k: int, seed: int) -> None:
    n = len(results)
    n_ok = sum(1 for r in results if r["correct"])

    cm = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in results:
        cm[r["true_class"]][r["predicted_class"]] += 1

    print("\n" + "=" * 62)
    print("STEP 6 SUMMARY -- 3 classes, balanced sample")
    print("=" * 62)
    print(f"Full dataset ({dist_full['total']:,} reviews): " +
          ", ".join(f"{c} {dist_full[c]:,} ({dist_full[c]/dist_full['total']:.1%})"
                    for c in CLASSES))
    print(f"Balanced sample: k={k} per class, seed={seed} -> "
          f"{', '.join(f'{c} {sample_counts[c]}' for c in CLASSES)} "
          f"({len(results)} total)")

    print(f"\nOverall accuracy: {n_ok}/{n} = {n_ok / n:.1%}")
    print("Per-class accuracy (true class -> correctly predicted):")
    for c in CLASSES:
        tot = sum(cm[c][p] for p in CLASSES)
        ok = cm[c][c]
        print(f"  {c:<8} {ok}/{tot} = {ok / tot:.1%}" if tot else f"  {c:<8} n/a")

    print("\nConfusion matrix (rows = true class, columns = predicted):")
    head = " " * 10 + "".join(f"{c:>10}" for c in CLASSES)
    print(head)
    for t in CLASSES:
        print(f"{t:<10}" + "".join(f"{cm[t][p]:>10}" for p in CLASSES))

    neu = cm["NEUTRAL"]
    print(f"\nHow 3-star NEUTRAL reviews are classified ({sample_counts['NEUTRAL']} total):")
    for p in CLASSES:
        print(f"  predicted {p:<8} {neu[p]:>3}  ({neu[p]/sample_counts['NEUTRAL']:.1%})"
              if sample_counts["NEUTRAL"] else f"  predicted {p:<8} n/a")

    # -------- comparison with Step 2's imbalanced binary run --------
    print("\n" + "-" * 62)
    print("Comparison: Step 2 (binary, imbalanced) vs Step 6 (3-class, balanced)")
    print("-" * 62)
    print(f"{'sample':<28} {'first 100 dataset lines':<28} "
          f"{'balanced, seed ' + str(seed):<28}")
    print(f"{'size / class mix':<28} {str(BINARY_REF['n']) + ' reviews, ' + BINARY_REF['mix']:<28} "
          f"{n} reviews, {sample_counts['POSITIVE']}:{sample_counts['NEUTRAL']}:{sample_counts['NEGATIVE']}")
    print(f"{'overall accuracy':<28} {BINARY_REF['overall']:<28} {n_ok/n:.1%}")
    for c in ("POSITIVE", "NEGATIVE"):
        tot = sum(cm[c][p] for p in CLASSES)
        ok = cm[c][c]
        print(f"{'accuracy ' + c.lower():<28} {BINARY_REF['per_class'][c]:<28} "
              f"{ok}/{tot} = {ok/tot:.1%}")
    print(f"{'NEUTRAL accuracy':<28} {'n/a (3* binned into NEGATIVE)':<28} "
          f"{cm['NEUTRAL']['NEUTRAL']}/{sample_counts['NEUTRAL']} = "
          f"{cm['NEUTRAL']['NEUTRAL']/sample_counts['NEUTRAL']:.1%}")

    print(f"\nResults saved to: {SAMPLE_PATH} and {RESULTS_PATH} "
          f"(step2_results.csv / step5_results.csv untouched)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=50,
                        help="reviews per class to sample (default 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the sample (default 42)")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "API key not found: create a .env file (see .env.example) or set OPENAI_API_KEY"
        )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not os.path.isfile(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found -- download it first (step2_classify.py)")

    print(f"Building balanced sample: k={args.k} per class, seed={args.seed}")
    sample, dist_full = build_sample(args.k, args.seed)
    sample_counts = {c: sum(1 for r in sample if r["class"] == c) for c in CLASSES}

    with open(SAMPLE_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset_index", "asin", "title", "text", "rating", "class"])
        for r in sample:
            w.writerow([r["dataset_index"], r["asin"], r["title"], r["text"],
                        r["rating"], r["class"]])

    print(f"Sample written: {SAMPLE_PATH} ({len(sample)} rows)")
    print(f"Model: {MODEL}\nEndpoint: {BASE_URL}/chat/completions\n"
          f"Rating is used ONLY for the true class; it is never sent to the LLM.\n")

    results = []
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as out_csv:
        w = csv.writer(out_csv)
        w.writerow(["dataset_index", "asin", "title", "text", "rating",
                    "true_class", "predicted_class", "correct", "llm_raw"])
        for i, s in enumerate(sample, 1):
            messages = build_3class_prompt(s["title"], s["text"])
            try:
                raw = chat(messages)
                predicted = parse_3class_label(raw)
            except Exception as e:
                raw = f"ERROR: {e}"
                predicted = "ERROR"
            correct = predicted == s["class"]
            row = {**s, "true_class": s["class"], "predicted_class": predicted,
                   "correct": correct, "llm_raw": raw}
            results.append(row)
            w.writerow([row[c] for c in
                        ("dataset_index", "asin", "title", "text", "rating",
                         "true_class", "predicted_class", "correct", "llm_raw")])
            out_csv.flush()
            print(f"  {i:3d}/{len(sample):3d} true={s['class']:<8} "
                  f"pred={predicted:<8} {'ok' if correct else 'MISMATCH'}",
                  flush=True)
            time.sleep(REQUEST_DELAY_S)

    report(results, dist_full, sample_counts, args.k, args.seed)


if __name__ == "__main__":
    main()
