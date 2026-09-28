"""MBAX 6418 - Assignment 1, Step 2: classify the first 100 Amazon 2023
Gift Cards reviews with the Step 1 prompt and measure agreement with the
rating-derived label. The LLM never sees the rating.

Pipeline
--------
  1. Download the Amazon 2023 Gift_Cards review dataset (gzipped JSON
     Lines) if it is not already in data/.
  2. Read the first N reviews (default 100): title, text, rating.
  3. Send ONLY title + text to the Step 1 classifier.
  4. Derive the correct label from the rating: rating >= 4 -> POSITIVE.
  5. Compare, and write one row per review to step2_results.csv.
  6. Print accuracy, per-class performance, and the mismatched reviews.

Usage
-----
    python step2_classify.py                 # download (if needed) + classify 100
    python step2_classify.py --limit 10      # smaller trial run
    python step2_classify.py --fields        # download + inspect fields, then exit

Output
------
    step2_results.csv -- index, asin, title, text, rating, correct_label,
                        predicted_label, correct, raw_output

Configuration comes from .env (gitignored) or real environment variables:
OPENAI_BASE_URL, OPENAI_API_KEY, MODEL. No credentials live in this file.
"""

import argparse
import csv
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

from review_prompt import build_prompt, parse_sentiment, load_env_file

N_REVIEWS_DEFAULT = 100
REQUEST_DELAY_S = 0.2  # be polite to the shared class endpoint

# The assignment sheet points at the UCSD McAuley datarepo mirror, which
# currently returns 404; the official McAuley-lab mirror on Hugging Face
# serves the identical raw file. First URL wins; the rest are fallbacks.
DATASET_URLS = [
    ("https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/"
     "review_categories/Gift_Cards.jsonl.gz", "gz"),
    ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
     "main/raw/review_categories/Gift_Cards.jsonl", "jsonl"),
]
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "Gift_Cards.jsonl.gz")
TMP_JSONL = os.path.join(DATA_DIR, "Gift_Cards.jsonl")
RESULTS_PATH = "step2_results.csv"

load_env_file()
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

FIELDS = [
    "index", "asin", "title", "text", "rating",
    "correct_label", "predicted_label", "correct", "raw_output",
]


def _download(url: str, dest: str) -> None:
    """Stream a URL to a local file, printing progress."""
    print(f"  downloading {url}", flush=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "MBAX6418-assignment (education)"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and done % (8 << 20) < (1 << 20):
                print(f"    {done/1e6:,.1f} / {total/1e6:,.1f} MB", flush=True)


def download_dataset() -> None:
    """Fetch the dataset once; subsequent runs reuse the local gzip copy."""
    if os.path.isfile(DATA_PATH):
        print(f"Dataset already present: {DATA_PATH} ({os.path.getsize(DATA_PATH)/1e6:,.1f} MB)")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    last_error = None
    for url, kind in DATASET_URLS:
        try:
            _download(url, TMP_JSONL)
            if kind == "gz":
                shutil.move(TMP_JSONL, DATA_PATH)
            else:
                print("  compressing to Gift_Cards.jsonl.gz ...", flush=True)
                with open(TMP_JSONL, "rb") as fin, gzip.open(DATA_PATH, "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 20)
                os.remove(TMP_JSONL)
            print(f"Saved {DATA_PATH} ({os.path.getsize(DATA_PATH)/1e6:,.1f} MB)")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_error = e
            print(f"  FAILED: {e}", flush=True)
            if os.path.exists(TMP_JSONL):
                os.remove(TMP_JSONL)
    raise SystemExit(f"Could not download the dataset from any mirror. Last error: {last_error}")


def inspect_dataset() -> int:
    """Stream the whole gzip file (proves it reads cleanly) and print fields."""
    n = 0
    first = None
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if first is None:
                first = json.loads(line)
            n += 1
    print(f"\nDataset: {DATA_PATH} -- {n:,} reviews total, gzip reads cleanly")
    for k, v in first.items():
        s = str(v)
        shown = s if len(s) <= 80 else s[:77] + "..."
        print(f"  {k:<18} {type(v).__name__:<8} {shown!r}")
    return n


def correct_label(rating: int) -> str:
    """Assignment rule: rating >= 4 is POSITIVE, otherwise NEGATIVE."""
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def chat(messages: list[dict], retries: int = 2) -> str:
    """Call the OpenAI-compatible endpoint; returns the assistant text."""
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


def report(results: list[dict]) -> None:
    n = len(results)
    n_ok = sum(r["correct"] for r in results)
    by_class = {c: [r for r in results if r["correct_label"] == c]
                for c in ("POSITIVE", "NEGATIVE")}

    print("\n" + "=" * 46)
    print("STEP 2 SUMMARY")
    print("=" * 46)
    print(f"Reviews classified: {n}")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {BASE_URL}/chat/completions")

    print(f"\nClass counts in the {n}-row sample (derived from rating):")
    for c in ("POSITIVE", "NEGATIVE"):
        print(f"  {c:<8} {len(by_class[c]):3d} reviews")
    print(f"  Rating distribution: "
          f"{dict(sorted(Counter(r['rating'] for r in results).items()))}")

    print(f"\nOverall accuracy: {n_ok}/{n} = {n_ok / n:.1%}")
    print(f"  correct: {n_ok}   incorrect: {n - n_ok}")
    for c in ("POSITIVE", "NEGATIVE"):
        sub = by_class[c]
        if not sub:
            print(f"  {c:<8} reviews: n/a (none in sample)")
            continue
        c_ok = sum(r["correct"] for r in sub)
        print(f"  {c:<8} reviews: {c_ok}/{len(sub)} correct = {c_ok / len(sub):.1%}")

    bad = [r for r in results if not r["correct"]]
    print(f"\nMismatched reviews ({len(bad)}):")
    for r in bad:
        print(f"\n  [{r['index']:3d}] rating={r['rating']}  "
              f"expected={r['correct_label']}  predicted={r['predicted_label']}")
        print(f"       title: {r['title'][:90]!r}")
        if r["text"]:
            print(f"       text : {r['text'][:140]!r}")
        if r["predicted_label"] in ("UNPARSEABLE", "ERROR"):
            print(f"       raw  : {r['raw_output'][:160]!r}")
    print(f"\nRaw results saved to: {RESULTS_PATH} ({n} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=N_REVIEWS_DEFAULT,
                        help=f"number of reviews to classify (default {N_REVIEWS_DEFAULT})")
    parser.add_argument("--fields", action="store_true",
                        help="download the dataset and inspect its fields, then exit")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "API key not found: create a .env file (see .env.example) or set OPENAI_API_KEY"
        )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    download_dataset()
    inspect_dataset()
    if args.fields:
        return

    print(f"\nClassifying the first {args.limit} reviews with model '{MODEL}'")
    print("Rating is used ONLY for evaluation afterwards; it is never sent to the model.\n")

    results = []
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(FIELDS)
        with gzip.open(DATA_PATH, "rt", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx >= args.limit:
                    break
                rec = json.loads(line)
                title = (rec.get("title") or "").strip()
                text = (rec.get("text") or "").strip()
                rating = rec.get("rating")
                if rating is None:
                    print(f"  [{idx}] skipping review without a rating", flush=True)
                    continue
                expected = correct_label(rating)

                # Step 1 prompt: ONLY title + text are sent to the model.
                messages = build_prompt(title, text)
                try:
                    raw = chat(messages)
                    try:
                        predicted = parse_sentiment(raw)
                    except ValueError:
                        predicted = "UNPARSEABLE"
                except Exception as e:
                    raw = f"ERROR: {e}"
                    predicted = "ERROR"

                ok = predicted == expected
                row = {
                    "index": idx, "asin": rec.get("asin", ""), "title": title,
                    "text": text, "rating": rating, "correct_label": expected,
                    "predicted_label": predicted, "correct": ok, "raw_output": raw,
                }
                results.append(row)
                writer.writerow([row[k] for k in FIELDS])
                out_csv.flush()  # partial progress survives a crash
                print(f"  {idx + 1:3d}/{args.limit:3d} "
                      f"expected={expected:<8} predicted={predicted:<8} "
                      f"{'ok' if ok else 'MISMATCH'}", flush=True)
                time.sleep(REQUEST_DELAY_S)

    report(results)


if __name__ == "__main__":
    main()
