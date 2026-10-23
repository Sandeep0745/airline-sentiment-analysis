"""Predict the sentiment of new tweets with the saved model.

Usage::

    python -m src.predict "@united my flight got cancelled again" "thanks for the great crew!"
    type tweets.txt | python -m src.predict      # one tweet per line from stdin
"""
import argparse
import sys

import pandas as pd

from src.config import FINAL_MODEL_PATH
from src.modeling import load_model


def predict_sentiment(texts, model):
    """Return a DataFrame with the predicted label and class probabilities for raw tweets."""
    texts = list(texts)
    probabilities = pd.DataFrame(model.predict_proba(texts), columns=model.classes_).round(3)
    probabilities.insert(0, "predicted", model.predict(texts))
    probabilities.insert(0, "text", texts)
    return probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", help="tweets to classify (reads stdin when omitted)")
    args = parser.parse_args(argv)

    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        parser.error("no text given")

    model = load_model(FINAL_MODEL_PATH)["model"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(predict_sentiment(texts, model).to_string(index=False))


if __name__ == "__main__":
    main()
