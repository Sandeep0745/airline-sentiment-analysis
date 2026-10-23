"""Evaluation helpers: test-set metrics, per-class reports and repeated-split evaluation."""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import RANDOM_STATE, SENTIMENT_LABELS
from src.data import split_train_test


def score_predictions(y_true, y_pred):
    """Accuracy, macro precision / recall / F1 and weighted F1."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def fit_and_score(name, model, X_train, y_train, X_test, y_test):
    """Fit on the training data, score once on the test data and return a result row."""
    model.fit(X_train, y_train)
    return {"model": name, **score_predictions(y_test, model.predict(X_test))}


def per_class_report(y_true, y_pred, labels=SENTIMENT_LABELS):
    """Precision, recall, F1 and support for every class as a DataFrame."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=labels)


def repeated_split_evaluation(df, build_model, param_grid, seeds=range(10), text_column="text",
                              target="airline_sentiment", inner_folds=3):
    """Evaluate a model family on several random 80/20 splits without tuning leakage.

    For every seed, hyperparameters in ``param_grid`` are re-selected by an inner
    stratified CV on **that seed's training split only**, the best model is refit on the
    training split and scored once on that seed's test split.
    """
    rows = []
    for seed in seeds:
        train_df, test_df = split_train_test(df, target=target, random_state=seed)
        search = GridSearchCV(
            clone(build_model()), param_grid, scoring="f1_macro", n_jobs=-1,
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE),
        )
        search.fit(train_df[text_column], train_df[target])
        predictions = search.best_estimator_.predict(test_df[text_column])
        rows.append({"seed": seed, **search.best_params_, **score_predictions(test_df[target], predictions)})
    return pd.DataFrame(rows)
