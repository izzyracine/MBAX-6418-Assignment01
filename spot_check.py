"""Spot-check the review prompt against obviously positive / negative reviews.

All test cases are deliberately unambiguous: if the model mislabels one of
these, the prompt (or the model choice) needs work before you scale up.

Configuration -- NEVER hardcode secrets in this file
----------------------------------------------------
Create a .env file in this folder (copy .env.example), and never commit it:
it holds the API key, and .gitignore already excludes it.

    OPENAI_API_KEY=your-key
    OPENAI_BASE_URL=https://professor-endpoint.example.com/v1
    MODEL=gpt-4o-mini

Real shell environment variables take precedence over .env, e.g.:

    export OPENAI_API_KEY=sk-...            # bash / git-bash
    $env:OPENAI_API_KEY = "sk-..."          # PowerShell
    python spot_check.py

No third-party packages required: uses the standard library only.
"""

import json
import os
import urllib.request

from review_prompt import build_prompt, load_env_file, parse_sentiment

load_env_file()

MODEL = os.environ.get("MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# (expected_label, title, text)
TEST_CASES = [
    ("POSITIVE", "Love them!",
     "These earbuds sound amazing and the battery lasts all week. Best "
     "purchase this year."),
    ("POSITIVE", "Five stars from me",
     "Works exactly as described. Shipping was fast and it feels really "
     "high quality."),
    ("POSITIVE", "Not bad for the price",
     "Does the job. Battery life is fine and it was cheap."),  # mild, but +ve
    ("NEGATIVE", "Broken after a week",
     "The product stopped working after seven days. Customer service never "
     "replied to my emails. Do not buy."),
    ("NEGATIVE", "Garbage",
     "Cheap plastic that fell apart immediately. Complete waste of money, "
     "I want a refund."),
    ("NEGATIVE", "Disappointed",
     "Expected better for this price, the sound is tinny and the build "
     "feels cheap."),  # mild, but clearly -ve
]


def chat(messages: list[dict]) -> str:
    """Call an OpenAI-compatible chat completions endpoint, return assistant text."""
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise SystemExit(f"API error {e.code} from {BASE_URL}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {BASE_URL}: {e.reason}")
    return data["choices"][0]["message"]["content"]


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "API key not found. Create a .env file (see .env.example) or set\n"
            "  OPENAI_API_KEY  (bash: export OPENAI_API_KEY=sk-...)"
        )

    print(f"Model: {MODEL}")
    print(f"Endpoint: {BASE_URL}/chat/completions")
    print("API key: set\n")

    passed = 0
    for expected, title, text in TEST_CASES:
        raw = chat(build_prompt(title, text))
        try:
            got = parse_sentiment(raw)
        except ValueError:
            got = f"UNPARSEABLE ({raw[:40]!r})"
        ok = got == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  expected={expected:<8} got={got:<12} {title!r}")

    print(f"\n{passed}/{len(TEST_CASES)} spot-checks passed")
    if passed < len(TEST_CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
