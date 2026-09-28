"""MBAX 6418 - Assignment 1, Step 5: primary-emotion detection.

Two independent emotion detectors per review (title + text only; the star
rating is never sent to the LLM):

  1. LLM: the updated prompt (review_prompt.build_emotion_prompt) asks for
     sentiment AND one primary emotion from the eight NRC categories.
  2. NRC Emotion Lexicon v0.92: each word in the review is looked up, every
     associated emotion scores +1, totals are summed per emotion, and the
     highest-scoring emotion is the primary emotion. Ties are broken by the
     fixed order of review_prompt.EMOTION_LIST. Reviews with no lexicon
     matches get "none".

Reviews and their Step 2 results are read from step2_results.csv -- nothing
is re-downloaded and Step 2 is not re-run. BOTH emotion results are kept per
review in a NEW file step5_results.csv (step2_results.csv stays untouched).
LLM-NRC agreement and disagreement examples are reported at the end.

Usage:
    python step5_emotions.py [--limit N]
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter

from review_prompt import (
    EMOTION_LIST,
    build_emotion_prompt,
    load_env_file,
    parse_emotion_output,
)

CSV_PATH = "step2_results.csv"
OUT_PATH = "step5_results.csv"
DATA_DIR = "data"
NRC_TXT = os.path.join(DATA_DIR, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
NRC_JSON = os.path.join(DATA_DIR, "nrc_en.json")
REQUEST_DELAY_S = 0.2  # be polite to the shared class endpoint

load_env_file()
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

FIELDS = ["index", "asin", "title", "text", "rating", "correct_label",
          "predicted_label", "llm_sentiment", "llm_emotion", "llm_raw",
          "nrc_matches", "nrc_scores", "nrc_emotion", "agree"]


# ---------------------------------------------------------------------------
# NRC Emotion Lexicon (word -> set of emotions)
# ---------------------------------------------------------------------------
def _load_txt(path: str) -> dict[str, set[str]]:
    """Canonical v0.92 format: word TAB emotion TAB 0|1 (one line per pair)."""
    lex: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            word, emo, assoc = parts
            if emo not in EMOTION_LIST or assoc != "1":
                continue
            lex.setdefault(word.strip().lower(), set()).add(emo)
    return lex


def _load_json(path: str) -> dict[str, set[str]]:
    """JSON form (nrc_en.json from the NRCLex package): word -> [emotions]."""
    data = json.load(open(path, encoding="utf-8"))
    lex: dict[str, set[str]] = {}
    for word, emos in data.items():
        hits = {e for e in emos if e in EMOTION_LIST}
        if hits:
            lex[str(word).lower()] = hits
    return lex


def ensure_lexicon() -> dict[str, set[str]]:
    """Return the lexicon; prefers the canonical .txt, else the local JSON.
    On first run with neither present, fetches the JSON from the official
    NRCLex PyPI wheel (a redistribution of NRC v0.92)."""
    if os.path.isfile(NRC_TXT):
        lex = _load_txt(NRC_TXT)
        print(f"NRC lexicon: {NRC_TXT} ({len(lex):,} words)")
        return lex
    if not os.path.isfile(NRC_JSON):
        print("NRC lexicon not present -- fetching nrc_en.json from the "
              "NRCLex PyPI wheel ...", flush=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        meta_req = urllib.request.Request(
            "https://pypi.org/pypi/nrclex/json",
            headers={"User-Agent": "curl/8"},
        )
        with urllib.request.urlopen(meta_req, timeout=30) as resp:
            meta = json.loads(resp.read().decode())
        wheel = next(u["url"] for u in meta["urls"]
                     if u["packagetype"] == "bdist_wheel")
        wheel_data = urllib.request.urlopen(
            urllib.request.Request(wheel, headers={"User-Agent": "curl/8"}),
            timeout=120,
        ).read()
        with zipfile.ZipFile(io.BytesIO(wheel_data)) as z:
            with open(NRC_JSON, "wb") as fh:
                fh.write(z.read("nrclex/data/nrc_en.json"))
    lex = _load_json(NRC_JSON)
    print(f"NRC lexicon: {NRC_JSON} ({len(lex):,} words)")
    return lex


def emotion_scores(text: str, lex: dict[str, set[str]]) -> tuple[dict[str, int], int]:
    """Total per-emotion scores for a review; returns (scores, words_matched)."""
    scores = {e: 0 for e in EMOTION_LIST}
    matched = 0
    for token in re.findall(r"[a-z']+", (text or "").lower()):
        emos = lex.get(token)
        if not emos:
            continue
        matched += 1
        for e in emos:
            scores[e] += 1
    return scores, matched


def primary_emotion(scores: dict[str, int]) -> str:
    """Highest-scoring emotion; ties broken by EMOTION_LIST order; 'none' if
    no lexicon words matched."""
    best = max(scores.values()) if scores else 0
    if best == 0:
        return "none"
    for e in EMOTION_LIST:
        if scores.get(e, 0) == best:
            return e


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(results: list[dict]) -> None:
    n = len(results)
    n_agree = sum(1 for r in results if r["agree"])

    print("\n" + "=" * 60)
    print("STEP 5 SUMMARY")
    print("=" * 60)
    print(f"Reviews processed: {n}")
    print(f"\nLLM vs NRC primary-emotion agreement: "
          f"{n_agree}/{n} = {n_agree / n:.1%}")

    print("\nLLM primary-emotion distribution:",
          dict(Counter(r["llm_emotion"] for r in results).most_common()))
    print("NRC primary-emotion distribution:",
          dict(Counter(r["nrc_emotion"] for r in results).most_common()))

    n_none = sum(1 for r in results if r["nrc_emotion"] == "none")
    n_unk = sum(1 for r in results if r["llm_emotion"] == "UNKNOWN")
    n_err = sum(1 for r in results if r["llm_emotion"] == "ERROR")
    print(f"\nNRC found no lexicon words in: {n_none} reviews | "
          f"LLM UNKNOWN emotion: {n_unk} | LLM call errors: {n_err}")

    drift = [r for r in results
             if r["llm_sentiment"] in ("POSITIVE", "NEGATIVE")
             and r["llm_sentiment"] != r["predicted_label"]]
    print(f"Sentiment: the new LLM pass agreed with Step 2 on "
          f"{n - len(drift)}/{n} reviews"
          f" ({len(drift)} moved)")

    disagree = [r for r in results if not r["agree"]]
    print(f"\nDisagreement examples ({len(disagree)} total, showing up to 15):")
    for r in disagree[:15]:
        top = sorted(json.loads(r["nrc_scores"]).items(),
                     key=lambda kv: -kv[1])[:3]
        top_s = ", ".join(f"{k}={v}" for k, v in top)
        print(f"\n  [{r['index']}] rating={r['rating']}  "
              f"LLM={r['llm_emotion']:<13} NRC={r['nrc_emotion']:<6} "
              f"(NRC top: {top_s})")
        print(f"      title: {r['title'][:80]!r}")
        print(f"      text : {r['text'][:110]!r}")

    print(f"\nResults saved to: {OUT_PATH} ({n} rows) -- "
          f"{CSV_PATH} untouched")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=100,
                        help="max reviews to process (default 100)")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(
            "API key not found: create a .env file (see .env.example) or set OPENAI_API_KEY"
        )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    lex = ensure_lexicon()

    with open(CSV_PATH, encoding="utf-8", newline="") as fh:
        rows_in = list(csv.DictReader(fh))[: args.limit]
    print(f"Using {len(rows_in)} saved reviews from {CSV_PATH} "
          f"(Step 2 results; not re-run)")
    print(f"Model: {MODEL}\nEndpoint: {BASE_URL}/chat/completions\n")

    results = []
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(FIELDS)
        for i, r in enumerate(rows_in, 1):
            title = (r["title"] or "").strip()
            text = (r["text"] or "").strip()

            # Rating is NEVER part of the prompt: title + text only.
            messages = build_emotion_prompt(title, text)
            try:
                raw = chat(messages)
                llm_sent, llm_emo = parse_emotion_output(raw)
            except Exception as e:
                raw = f"ERROR: {e}"
                llm_sent, llm_emo = "ERROR", "ERROR"

            scores, matched = emotion_scores(title + " " + text, lex)
            nrc_emo = primary_emotion(scores)
            agree = (llm_emo == nrc_emo
                     and llm_emo not in ("UNKNOWN", "ERROR")
                     and nrc_emo != "none")

            row = {
                "index": r["index"], "asin": r["asin"], "title": title,
                "text": text, "rating": r["rating"],
                "correct_label": r["correct_label"],
                "predicted_label": r["predicted_label"],
                "llm_sentiment": llm_sent, "llm_emotion": llm_emo,
                "llm_raw": raw, "nrc_matches": matched,
                "nrc_scores": json.dumps(scores), "nrc_emotion": nrc_emo,
                "agree": agree,
            }
            results.append(row)
            writer.writerow([row[k] for k in FIELDS])
            out_csv.flush()  # partial progress survives a crash
            print(f"  {i:3d}/{len(rows_in):3d} llm={llm_emo:<13} "
                  f"nrc={nrc_emo:<6} {'AGREE' if agree else 'disagree'}",
                  flush=True)
            time.sleep(REQUEST_DELAY_S)

    report(results)


if __name__ == "__main__":
    main()
