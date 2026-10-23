"""Data loading, the airline-label correction rule and the shared train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLEAN_DATA_PATH, RANDOM_STATE, RAW_DATA_PATH, TEST_SIZE

MISSING_DATA_MESSAGE = (
    "{path} not found.\n"
    "Download Tweets.csv from Kaggle (see README > Dataset) into data/raw/, "
    "then run notebooks/02_data_cleaning.ipynb to create the processed file."
)


def load_raw_tweets(path=RAW_DATA_PATH):
    """Load the original Kaggle file exactly as provided."""
    if not path.exists():
        raise FileNotFoundError(MISSING_DATA_MESSAGE.format(path=path))
    return pd.read_csv(path)


def load_clean_tweets(path=CLEAN_DATA_PATH, parse_dates=False):
    """Load the cleaned dataset written by notebook 02."""
    if not path.exists():
        raise FileNotFoundError(MISSING_DATA_MESSAGE.format(path=path))
    return pd.read_csv(
        path,
        dtype={"tweet_id": "string"},
        parse_dates=["tweet_created", "tweet_date"] if parse_dates else None,
    )


def correct_airline_labels(df, text_column="text", airline_column="airline"):
    """Relabel the source's ``Delta`` rows that are demonstrably JetBlue tweets.

    In the source file, the rows labelled ``Delta`` are almost all addressed to
    ``@JetBlue`` (every other airline's rows mention that airline's own handle).
    Rule: a row labelled ``Delta`` whose text mentions ``@jetblue`` becomes ``JetBlue``.
    Other ``Delta`` rows are left unchanged and flagged with ``airline_verified = False``
    so the caller can decide what to do with them.

    Returns a copy with the corrected ``airline`` column, the original label in
    ``airline_source`` and a boolean ``airline_verified`` column.
    """
    out = df.copy()
    text = out[text_column].fillna("").str.lower()
    labelled_delta = out[airline_column] == "Delta"
    mentions_jetblue = text.str.contains("@jetblue", regex=False)

    out["airline_source"] = out[airline_column]
    out.loc[labelled_delta & mentions_jetblue, airline_column] = "JetBlue"
    out["airline_verified"] = ~(labelled_delta & ~mentions_jetblue)
    return out


def split_train_test(df, target="airline_sentiment", random_state=RANDOM_STATE, test_size=TEST_SIZE):
    """Stratified train/test split used by every text model in the project."""
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
