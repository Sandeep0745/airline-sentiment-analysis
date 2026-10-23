"""Train the final sentiment model and save it to ``models/``.

Usage::

    python -m src.train            # train on the 80% split, report test metrics, save
    python -m src.train --all-data # same evaluation, then refit on all rows before saving
"""
import argparse
import json

from src.config import FINAL_MODEL_PATH, SENTIMENT_LABELS
from src.data import load_clean_tweets, split_train_test
from src.evaluation import per_class_report, score_predictions
from src.modeling import build_final_model, save_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-data", action="store_true",
                        help="after evaluation, refit on every row before saving")
    args = parser.parse_args(argv)

    df = load_clean_tweets()
    train_df, test_df = split_train_test(df)
    print(f"Training on {len(train_df):,} tweets, testing on {len(test_df):,}")

    model = build_final_model().fit(train_df["text"], train_df["airline_sentiment"])
    predictions = model.predict(test_df["text"])
    metrics = score_predictions(test_df["airline_sentiment"], predictions)

    print("\nHeld-out test set")
    for name, value in metrics.items():
        print(f"  {name:<16} {value:.4f}")
    print("\n" + per_class_report(test_df["airline_sentiment"], predictions, SENTIMENT_LABELS).round(3).to_string())

    if args.all_data:
        model = build_final_model().fit(df["text"], df["airline_sentiment"])
        metrics["note"] = "metrics measured on the 80/20 split; saved model refit on all rows"

    path = save_model(model, FINAL_MODEL_PATH, metrics=metrics)
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
