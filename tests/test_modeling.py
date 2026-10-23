import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from src.data import correct_airline_labels, split_train_test
from src.evaluation import per_class_report, score_predictions
from src.modeling import (
    NBLogisticRegression, build_final_model, build_word_char_lr, build_word_lr, load_model, save_model,
)
from src.predict import predict_sentiment

TEMPLATES = {
    "Negative": ["flight delayed again worst service", "lost my bag terrible", "cancelled flight no help at all",
                 "not happy, on hold for hours", "rude staff and late departure"],
    "Neutral": ["what is the baggage allowance?", "can you check my flight status?", "do you fly to boston?",
                "is there wifi on board?", "how do i change my seat?"],
    "Positive": ["thanks for the great crew", "amazing flight love it :)", "best service ever thank you",
                 "awesome landing, thanks!", "great job on time"],
}


@pytest.fixture(scope="module")
def toy_tweets():
    rows = [{"text": f"@airline {text} {i}", "airline_sentiment": label}
            for label, texts in TEMPLATES.items() for i in range(8) for text in texts]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("builder", [build_word_lr, build_word_char_lr, build_final_model])
def test_pipelines_fit_and_predict_raw_text(builder, toy_tweets):
    model = builder().fit(toy_tweets["text"], toy_tweets["airline_sentiment"])
    predictions = model.predict(["@united thanks for the amazing crew", "@united flight delayed, lost bag"])
    assert list(predictions) == ["Positive", "Negative"]


def test_nb_logistic_regression_probabilities(toy_tweets):
    model = build_final_model().fit(toy_tweets["text"], toy_tweets["airline_sentiment"])
    probabilities = model.predict_proba(toy_tweets["text"][:5])
    assert probabilities.shape == (5, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert list(model.classes_) == ["Negative", "Neutral", "Positive"]
    # predict must agree with the most probable class
    assert (model.classes_[probabilities.argmax(axis=1)] == model.predict(toy_tweets["text"][:5])).all()


def test_nb_logistic_regression_is_sklearn_compatible(toy_tweets):
    estimator = NBLogisticRegression(C=2)
    assert clone(estimator).get_params()["C"] == 2
    scores = cross_val_score(build_final_model(), toy_tweets["text"], toy_tweets["airline_sentiment"], cv=3)
    assert scores.mean() > 0.8


def test_save_load_and_predict_roundtrip(tmp_path, toy_tweets):
    model = build_final_model().fit(toy_tweets["text"], toy_tweets["airline_sentiment"])
    path = save_model(model, tmp_path / "model.joblib", metrics={"accuracy": 1.0})
    bundle = load_model(path)
    assert bundle["metrics"] == {"accuracy": 1.0}
    result = predict_sentiment(["thanks for the great crew"], bundle["model"])
    assert list(result.columns) == ["text", "predicted", "Negative", "Neutral", "Positive"]
    assert result.loc[0, "predicted"] == "Positive"


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_score_predictions_and_report():
    y_true = ["Negative", "Neutral", "Positive", "Negative"]
    y_pred = ["Negative", "Negative", "Positive", "Negative"]
    scores = score_predictions(y_true, y_pred)
    assert scores["accuracy"] == pytest.approx(0.75)
    report = per_class_report(y_true, y_pred)
    assert report.loc["Neutral", "recall"] == 0
    assert report["support"].sum() == 4


def test_correct_airline_labels():
    df = pd.DataFrame({
        "airline": ["Delta", "Delta", "Delta", "United"],
        "text": ["@JetBlue late again", "@DeltaAssist help", "We think it's a treat", "@united @JetBlue hi"],
    })
    out = correct_airline_labels(df)
    assert list(out["airline"]) == ["JetBlue", "Delta", "Delta", "United"]
    assert list(out["airline_source"]) == ["Delta", "Delta", "Delta", "United"]
    assert list(out["airline_verified"]) == [True, False, False, True]
    assert list(df["airline"]) == ["Delta", "Delta", "Delta", "United"]  # input not modified


def test_split_is_stratified(toy_tweets):
    train, test = split_train_test(toy_tweets)
    assert len(test) == round(len(toy_tweets) * 0.2)
    shares = test["airline_sentiment"].value_counts(normalize=True)
    assert shares.max() - shares.min() < 0.05
