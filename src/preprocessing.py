"""Tweet text cleaning.

Two cleaners are provided:

* ``clean_tweet``      - light normalisation used by the baseline models (notebooks 06-08).
* ``clean_tweet_plus`` - sentiment-aware cleaning used by the final model (notebook 10):
  emoticon and emoji tokens, elongation shortening and negation-scope marking.

Order matters in ``clean_tweet_plus``: URLs and @mentions are removed *before*
emoticons are detected, otherwise the ``:/`` in ``http://`` is read as a frown.
"""
import html
import re

import pandas as pd

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S*|\bhttps?\b:?", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})

# Emoticons are matched on the original-case text (":D", ":P").
# Letter-ending emoticons need non-word boundaries so "re:delay" or ":do" are not matched.
EMOTICON_PATTERNS = [
    (re.compile(r"[:;=]-?[)\]]|(?<!\w)[:;=]-?[dp](?!\w)|<3(?!\d)", re.IGNORECASE), " emo_pos "),
    (re.compile(r"[:;=]-?[(\[|]|(?<!\w)[:;=]-?[/\\](?![/\w])"), " emo_neg "),
]
EMOJI_PATTERN = re.compile("([\U0001F300-\U0001FAFF☀-➿])")
EMOJI_JOINERS = re.compile("[️‍]")
ELONGATION_PATTERN = re.compile(r"(\w)\1{2,}")
TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]")
WORD_TOKEN = re.compile(r"[\w']+")
CLAUSE_BREAK = re.compile(r"[.!?,;]")

NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "none", "nowhere", "neither", "nor",
    "cannot", "cant", "dont", "didnt", "doesnt", "isnt", "wasnt", "wont", "wouldnt",
    "couldnt", "shouldnt", "arent", "havent", "hasnt", "aint",
}

# Handles of the airlines in the dataset; kept when masking user handles for display
AIRLINE_HANDLES = {
    "virginamerica", "united", "southwestair", "jetblue", "delta", "deltaassist",
    "usairways", "americanair",
}


def _is_missing(text):
    return text is None or (not isinstance(text, str) and pd.isna(text))


def _normalise(text):
    """Shared first steps: HTML entities, curly apostrophes, URLs and @mentions."""
    text = html.unescape(str(text)).translate(APOSTROPHES)
    text = URL_PATTERN.sub(" ", text)
    return MENTION_PATTERN.sub(" ", text)


def clean_tweet(text):
    """Light normalisation: decode HTML, normalise apostrophes, lower-case, drop URLs and @mentions."""
    if _is_missing(text):
        return ""
    text = _normalise(text).lower()
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_negation(token):
    return token in NEGATION_WORDS or token.endswith("n't")


def mark_negations(text, window=3):
    """Append ``_neg`` to up to ``window`` words after a negation, stopping at punctuation."""
    tokens, remaining = [], 0
    for token in TOKEN_PATTERN.findall(text):
        if is_negation(token):
            remaining = window
            tokens.append(token)
        elif CLAUSE_BREAK.fullmatch(token):
            remaining = 0
            tokens.append(token)
        elif WORD_TOKEN.fullmatch(token):
            if remaining > 0 and not token.startswith("emo_"):
                token += "_neg"
            tokens.append(token)
            remaining = max(0, remaining - 1)
        else:  # emoji and other symbols are kept as-is and do not use up the window
            tokens.append(token)
    return " ".join(tokens)


def clean_tweet_plus(text, negation_window=3):
    """Sentiment-aware cleaning used by the final model.

    >>> clean_tweet_plus("@united NOT happy :( sooooo late http://t.co/x")
    'not happy_neg emo_neg soo_neg late'
    """
    if _is_missing(text):
        return ""
    text = _normalise(text)
    for pattern, token in EMOTICON_PATTERNS:
        text = pattern.sub(token, text)
    text = EMOJI_JOINERS.sub("", text)
    text = EMOJI_PATTERN.sub(r" \1 ", text)
    text = text.lower()
    text = ELONGATION_PATTERN.sub(r"\1\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return mark_negations(text, window=negation_window)


def clean_texts(texts):
    """List version of ``clean_tweet`` for use inside a scikit-learn ``FunctionTransformer``."""
    return [clean_tweet(t) for t in texts]


def clean_texts_plus(texts):
    """List version of ``clean_tweet_plus`` for use inside a scikit-learn ``FunctionTransformer``."""
    return [clean_tweet_plus(t) for t in texts]


def mask_user_handles(text):
    """Replace non-airline @handles with ``@user`` so displayed tweets do not expose usernames."""
    if _is_missing(text):
        return ""
    return MENTION_PATTERN.sub(
        lambda m: m.group(0) if m.group(0)[1:].lower() in AIRLINE_HANDLES else "@user", str(text)
    )
