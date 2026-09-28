"""Reusable structured prompt for Amazon review sentiment classification.

MBAX 6418, Assignment 1, Step 1.

Design decisions
----------------
* The star rating is NEVER part of the input. The prompt builder only ever
  receives the review title + text, so the rating physically cannot leak
  into what the model sees (the leak risk is only if the review *text*
  itself mentions stars, which is inherent to the data).
* The model is told explicitly that it has no rating and must not assume one,
  so it classifies purely on the language of the review.
* Output is a single token (POSITIVE or NEGATIVE) so Python can read it with
  a plain string comparison -- no JSON parsing, no regex of prose.

Usage
-----
    from review_prompt import build_prompt, parse_sentiment, load_env_file

    messages = build_prompt("Love them!", "Best earbuds ever, battery lasts a week.")
    # -> [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    # send `messages` to any chat-completions API, then:
    label = parse_sentiment(model_reply)   # -> "POSITIVE" or "NEGATIVE"

    load_env_file() reads OPENAI_BASE_URL / OPENAI_API_KEY / MODEL from a
    gitignored .env file (or leaves real shell environment variables as-is),
    so API credentials never have to live in Python source code.

To use the prompt outside code (e.g. pasting into ChatGPT): paste the
SYSTEM_PROMPT as the system message and a formatted user message below it.
"""

import json
import os
import re

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon product reviews. "
    "You will receive a review's title and its text, and you must decide "
    "whether the overall sentiment expressed by the reviewer is positive or "
    "negative. You do NOT receive the star rating and you must not assume "
    "one. Base your decision only on the language of the review itself. "
    'Respond with exactly one word: "POSITIVE" or "NEGATIVE". '
    "Do not add explanations, quotes, punctuation, or any other text."
)

USER_PROMPT_TEMPLATE = (
    "Title: {title}\n"
    "Review: {text}\n"
    "\n"
    "Classify this review. Reply with only POSITIVE or NEGATIVE."
)


def build_prompt(title: str, text: str) -> list[dict]:
    """Build the chat-completions message list for one review.

    Returns [{"role": "system", ...}, {"role": "user", ...}] -- exactly the
    shape the OpenAI (and most compatible) APIs expect. No rating is ever
    passed in.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                title=title.strip(), text=text.strip()
            ),
        },
    ]


def parse_sentiment(response_text: str) -> str:
    """Extract 'POSITIVE' or 'NEGATIVE' from a model reply.

    Tolerates the sloppy output models occasionally produce: whitespace,
    surrounding quotes, trailing punctuation, lowercase, or a JSON wrapper
    such as {"sentiment": "negative"}. Raises ValueError if no label is found.
    """
    import re

    cleaned = response_text.strip().strip('"\'`').strip()
    if cleaned.upper() in ("POSITIVE", "NEGATIVE"):
        return cleaned.upper()

    match = re.search(r"\b(POSITIVE|NEGATIVE)\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    raise ValueError(
        f"Could not parse sentiment from model output: {response_text!r}"
    )


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (stdlib dotenv).

    Environment variables already set in the shell always win; a missing
    .env file is a no-op. Keeps API keys out of Python source files and out
    of git history.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


EMOTION_LIST = ("anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust")

# Small fallback for off-list answers; exact-token matches only.
EMOTION_SYNONYMS = {
    "joy": ("love", "loved", "happy", "glad", "delighted", "wonderful"),
    "anger": ("angry", "mad", "hate", "furious", "annoyed"),
    "fear": ("scared", "afraid", "fearful"),
    "sadness": ("sad", "sadly", "disappointed", "upset"),
    "anticipation": ("excited", "excitement", "eager"),
    "disgust": ("disgusted", "disgusting", "gross"),
    "surprise": ("surprised", "shocked"),
}

SYSTEM_PROMPT_EMOTION = (
    "You are an emotion and sentiment classifier for Amazon product "
    "reviews. You will receive a review's title and its text, and you must "
    "decide the overall sentiment (POSITIVE or NEGATIVE) and the single "
    "primary emotion expressed by the reviewer. You do NOT receive the star "
    "rating and you must not assume one; base your decision only on the "
    "language of the review itself. Choose the primary emotion from exactly "
    "this list: anger, anticipation, disgust, fear, joy, sadness, surprise, "
    "trust. Reply on one line as: SENTIMENT|emotion (for example "
    "POSITIVE|joy). Do not add explanations, quotes, or any other text."
)

USER_PROMPT_TEMPLATE_EMOTION = (
    "Title: {title}\n"
    "Review: {text}\n"
    "\n"
    "Reply with the sentiment and primary emotion as: SENTIMENT|emotion."
)


def build_emotion_prompt(title: str, text: str) -> list[dict]:
    """Like build_prompt, but asks for sentiment AND one primary emotion.

    The star rating is still never part of the input -- only title + text.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_EMOTION},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE_EMOTION.format(
                title=title.strip(), text=text.strip()
            ),
        },
    ]


def parse_emotion_output(response_text: str) -> tuple[str, str]:
    """Parse 'SENTIMENT|emotion' (or a JSON object) model output.

    Returns (sentiment, emotion): sentiment is POSITIVE/NEGATIVE/UNKNOWN,
    emotion is one of EMOTION_LIST or UNKNOWN. Emotion tokens outside the
    list are mapped through EMOTION_SYNONYMS; everything else is UNKNOWN.
    """
    txt = response_text.strip()

    # Tolerant JSON: {"sentiment": "POSITIVE", "emotion": "joy"}
    m = re.search(r"\{[^{}]*\}", txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = str(obj.get("sentiment", "")).upper()
            e = str(obj.get("emotion", "")).lower()
            if s in ("POSITIVE", "NEGATIVE") and e in EMOTION_LIST:
                return s, e
        except (ValueError, TypeError):
            pass

    s_match = re.search(r"\b(POSITIVE|NEGATIVE)\b", txt, re.IGNORECASE)
    sentiment = s_match.group(1).upper() if s_match else "UNKNOWN"

    low = txt.lower()
    for emo in EMOTION_LIST:
        if re.search(rf"\b{emo}\b", low):
            return sentiment, emo
    for emo, syns in EMOTION_SYNONYMS.items():
        for syn in syns:
            if re.search(rf"\b{syn}\b", low):
                return sentiment, emo
    return sentiment, "UNKNOWN"


SYSTEM_PROMPT_3CLASS = (
    "You are a sentiment classifier for Amazon product reviews. "
    "You will receive a review's title and its text, and you must decide "
    "whether the overall sentiment expressed by the reviewer is positive, "
    "neutral, or negative. You do NOT receive the star rating and you must "
    "not assume one. Base your decision only on the language of the review "
    'itself. Respond with exactly one word: "POSITIVE", "NEUTRAL", or '
    '"NEGATIVE". Do not add explanations, quotes, punctuation, or any '
    "other text."
)

USER_PROMPT_TEMPLATE_3CLASS = (
    "Title: {title}\n"
    "Review: {text}\n"
    "\n"
    "Classify this review. Reply with only POSITIVE, NEUTRAL, or NEGATIVE."
)


def build_3class_prompt(title: str, text: str) -> list[dict]:
    """Three-class variant: POSITIVE / NEUTRAL / NEGATIVE.

    The star rating is still never part of the input -- only title + text.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_3CLASS},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE_3CLASS.format(
                title=title.strip(), text=text.strip()
            ),
        },
    ]


def parse_3class_label(response_text: str) -> str:
    """Extract POSITIVE / NEUTRAL / NEGATIVE from a model reply.

    Tolerates whitespace, surrounding quotes, trailing punctuation,
    lowercase, and JSON wrappers such as {"sentiment": "neutral"}.
    Raises ValueError if no 3-class label is found.
    """
    txt = response_text.strip()
    m = re.search(r"\{[^{}]*\}", txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = str(obj.get("sentiment", "")).upper()
            if s in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
                return s
        except (ValueError, TypeError):
            pass
    match = re.search(r"\b(POSITIVE|NEUTRAL|NEGATIVE)\b", txt, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    raise ValueError(f"Could not parse 3-class label from: {response_text!r}")
