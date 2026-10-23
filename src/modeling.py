"""Model pipelines, the NB-weighted Logistic Regression and model persistence.

Every pipeline takes **raw tweet text** and cleans it as its first step, so the same
object can be cross-validated, evaluated, saved and used for inference.
"""
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import FINAL_MODEL_PATH, RANDOM_STATE
from src.features import final_tfidf, word_char_tfidf, word_tfidf
from src.preprocessing import clean_texts, clean_texts_plus

FINAL_C = 4  # selected in notebook 10 by 5-fold CV Macro-F1 on the training split


def balanced_logistic_regression(C=1.0, max_iter=2000):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=RANDOM_STATE)


def build_word_lr(C=1.0, ngram_range=(1, 2), class_weight="balanced"):
    """Baseline: basic cleaning -> word TF-IDF -> Logistic Regression (notebooks 06-07)."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts)),
        ("tfidf", word_tfidf(ngram_range)),
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_word_char_lr(C=2, word_ngram_range=(1, 2)):
    """Notebook 08 model: basic cleaning -> word + char TF-IDF -> Logistic Regression (C selected in notebook 08)."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts)),
        ("features", word_char_tfidf(word_ngram_range)),
        ("model", balanced_logistic_regression(C)),
    ])


class NBLogisticRegression(ClassifierMixin, BaseEstimator):
    """Naive-Bayes-weighted Logistic Regression (one-vs-rest).

    Follows the idea of NBSVM (Wang & Manning, 2012): for each class, every feature is
    scaled by its Naive Bayes log-count ratio ``r = log(p(f | class) / p(f | rest))`` and a
    linear classifier is trained on the scaled features.

    Deliberate differences from the original NBSVM:

    * ``r`` is computed from summed **TF-IDF weights** of the input matrix, not from
      binarised counts (the original uses binary features);
    * the classifier is a class-balanced **Logistic Regression** instead of an SVM, so
      the model outputs probabilities;
    * there is no interpolation between the NB and SVM weights (the original's beta);
    * multi-class is handled one-vs-rest, and ``predict_proba`` normalises the
      one-vs-rest probabilities to sum to 1.

    The ratios are learned in ``fit`` only, so inside a ``Pipeline`` they never see test data.
    """

    def __init__(self, C=FINAL_C, alpha=1.0, max_iter=3000):
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.ratios_, self.models_ = [], []
        for label in self.classes_:
            in_class = self.alpha + X[y == label].sum(axis=0)
            out_class = self.alpha + X[y != label].sum(axis=0)
            ratio = np.asarray(np.log((in_class / in_class.sum()) / (out_class / out_class.sum()))).ravel()
            model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=self.max_iter,
                                       random_state=RANDOM_STATE)
            model.fit(X.multiply(ratio).tocsr(), (y == label).astype(int))
            self.ratios_.append(ratio)
            self.models_.append(model)
        return self

    def decision_function(self, X):
        return np.column_stack([m.decision_function(X.multiply(r).tocsr()) for m, r in zip(self.models_, self.ratios_)])

    def predict_proba(self, X):
        scores = np.column_stack(
            [m.predict_proba(X.multiply(r).tocsr())[:, 1] for m, r in zip(self.models_, self.ratios_)]
        )
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def build_final_model(C=FINAL_C):
    """Final model (notebook 10): sentiment-aware cleaning -> final TF-IDF -> NB-weighted LR."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts_plus)),
        ("features", final_tfidf()),
        ("model", NBLogisticRegression(C=C)),
    ])


def save_model(model, path=FINAL_MODEL_PATH, metrics=None):
    """Save a fitted pipeline together with basic provenance information."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "metrics": metrics or {},
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, path)
    return path


def load_model(path=FINAL_MODEL_PATH):
    """Load a pipeline saved with ``save_model``. Returns the bundle dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first with: python -m src.train")
    return joblib.load(path)
