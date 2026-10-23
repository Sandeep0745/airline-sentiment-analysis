# Methodology

This document records the analytical decisions behind the project: what was done, why, and what the limits are. Results are in the notebooks and summarised in the [README](../README.md).

## 1. Data

**Source:** *Twitter US Airline Sentiment* (Figure Eight / CrowdFlower "Data for Everyone", distributed on Kaggle). It has 14,640 tweets about six US airlines, collected 16–24 February 2015. Contributors labelled each tweet as positive, neutral or negative, and gave negative tweets a complaint category.

### Quality issues found (notebook 01)

| Issue | Evidence | Treatment (notebook 02) |
|---|---|---|
| Near-empty columns | `airline_sentiment_gold` 99.7%, `negativereason_gold` 99.8%, `tweet_coord` 93.0% missing | Dropped |
| Personal identifier | `name` = poster's username | Dropped; @usernames in displayed tweets are masked as `@user` |
| Exact duplicates | 36 rows in the raw file, 39 after dropping the columns above | Removed |
| Repeated `tweet_id` with different labels | 155 extra rows; 18 IDs carry conflicting sentiment labels | Kept the record with the highest label confidence |
| Structural missing values | `negativereason` is missing only for non-negative tweets | Filled with `Not Applicable` (no rows dropped) |
| **Mislabelled airline** | Rows labelled "Delta": 2,218 of 2,222 address @JetBlue; every other airline's rows mention its own handle 100% of the time | Relabelled `JetBlue` by rule; original kept in `airline_source`; 4 unverifiable rows excluded |
| Uneven collection | 99.85% of American Airlines tweets fall on 22–24 Feb | Documented; time results treated as descriptive only |

### The Delta → JetBlue correction

The rule lives in `src.data.correct_airline_labels`:

1. A row labelled `Delta` whose text mentions `@jetblue` is relabelled `JetBlue`.
2. The original label is kept in `airline_source`, so the change is transparent and reversible.
3. Rows labelled `Delta` that do not mention `@jetblue` are flagged as unverified. There are 4 of them: two addressed to @DeltaAssist and two outbound replies with no airline handle. They are excluded because a two-tweet "Delta" group cannot support airline comparisons, and it would also break the expected-count assumption of the chi-square test.

Two pieces of evidence support the correction. First, the handle check above. Second, around 150 of these rows react to JetBlue's widely shared "on fleek" tweet from February 2015.

After cleaning, 14,481 rows remain (98.9% of the raw file), and an assertion confirms that every airline's tweets mention that airline's handle.

## 2. Descriptive and statistical analysis (notebooks 03–05)

- **Net Sentiment Score** = Positive % − Negative %. This is a project-defined KPI for comparing airlines; it ranges from −100 to +100. The mean of `sentiment_score` equals NSS ÷ 100.
- Airlines are compared on **within-airline percentages**, not raw counts, because tweet volume differs about 7.6× between airlines.
- Complaint leadership is reported both **by count** and **as a rate per 100 tweets**.
- The **chi-square test** of independence (airline × sentiment) is reported with **Cramér's V**, because with roughly 14,500 rows even small differences are "significant".
- **Bootstrap 95% confidence intervals** (2,000 resamples, fixed seed) are computed for each airline's Negative rate.
- **Kruskal–Wallis** compares retweet counts by sentiment. ANOVA is not used because retweets are zero-inflated and highly skewed. The effect size is reported as ε².

## 3. Text classification (notebooks 06–08, 10)

### Evaluation protocol

- **Target:** the supplied 3-class sentiment label. Model predictions never replace labels in the descriptive analysis.
- **Split:** a stratified 80/20 split with `random_state=42` (`src.data.split_train_test`), shared by all text models.
- **No test-set leakage.** Every cleaning step, vectoriser and classifier sits inside a scikit-learn `Pipeline`. These are fitted on training data only: inside cross-validation folds during model selection, and on the full training split for the final evaluation.
- **Model selection** uses stratified cross-validation on the training split only. Notebook 07 uses a 3-fold grid search. Notebooks 08 and 10 use 5-fold comparisons plus 3-fold grid searches. The held-out test set is scored once, after selection.
- **Metrics:** Macro-F1 is the primary metric, because the classes are imbalanced (≈63% Negative and a majority-class baseline already reaches ≈63% accuracy). Accuracy, weighted-F1, macro precision and recall, per-class precision/recall/F1 and confusion matrices are also reported.
- **Cross-validation scores and held-out test scores are always labelled separately.**

### Stability without tuning leakage

Earlier versions re-used hyperparameters chosen on the seed-42 training split when evaluating ten other random splits. That training split overlaps the other splits' test sets, so the reported spread was slightly optimistic.

`src.evaluation.repeated_split_evaluation` fixes this. For every seed, it re-selects `C` with a 3-fold grid search **inside that seed's training split**, refits, and scores once on that seed's test split. The feature design of each model family stays fixed, as an a-priori design choice.

### Preprocessing

`clean_tweet` (baselines) does the following:
- decodes HTML entities;
- normalises curly apostrophes;
- lower-cases the text;
- removes URLs and @mentions. Handles identify the airline, not the sentiment; an ablation in notebook 08 confirms that removing them does not hurt performance.

`clean_tweet_plus` (final model) adds:
- emoticons converted to `emo_pos` / `emo_neg` tokens, with word-boundary checks for letter emoticons such as `:D` and `:P`;
- each emoji as its own token;
- elongated words shortened (`sooooo` → `soo`);
- **negation scope marking**: up to three words after a negation get a `_neg` suffix, stopping at punctuation. Negations include contractions with straight or curly apostrophes and common apostrophe-less forms (`dont`, `cant`).

**Bug fix.** The earlier implementation replaced emoticons *before* removing URLs. As a result, the `:/` in every `http://` became a false `emo_neg` token, and fragments such as `t.co` leaked into the features (about 8% of tweets contain a URL). URLs and mentions are now removed first. Unit tests in `tests/test_preprocessing.py` cover URLs, emoticons, false emoticons (`re:delay`, `10:30`, `<30`), emoji, negation, contractions, mentions, empty strings and missing values. Notebook 10 §1.1 measures the effect of the fix.

### Models

| Notebook | Model |
|---|---|
| 06 | Dummy (majority class), Multinomial Naive Bayes, word 1–2-gram TF-IDF + class-balanced Logistic Regression |
| 07 | Grid search over n-grams, `C` and class weighting; coefficient-based interpretation; error analysis by annotator agreement |
| 08 | Word (1–2-gram, ≤40k) + character (3–5-gram, ≤50k) TF-IDF + Logistic Regression |
| 10 | `clean_tweet_plus` + punctuation-aware word TF-IDF + character 2–6-gram TF-IDF (no caps) + **NB-weighted Logistic Regression** |

### NB-weighted Logistic Regression vs NBSVM

`src.modeling.NBLogisticRegression` follows the idea of NBSVM (Wang & Manning, 2012): for each class, every feature is scaled by its Naive Bayes log-count ratio before a linear classifier is trained. The deliberate differences are:

| | Original NBSVM | This project |
|---|---|---|
| Features used for the ratio | binarised counts | summed TF-IDF weights |
| Classifier | linear SVM | class-balanced Logistic Regression (gives probabilities) |
| NB/SVM weight interpolation (β) | yes | no |
| Multi-class | binary | one-vs-rest; probabilities normalised to sum to 1 |

The ratios are learned in `fit`, so inside a pipeline they never see validation or test data.

## 4. Metadata model (notebook 09)

This model predicts `is_negative` from non-text features only: airline, user timezone, hour, retweet count and profile-completeness flags. Label-derived columns (`negativereason`, label confidence, sentiment flags) and calendar columns that only identify collection days are excluded. The timezone vocabulary (top 10 + "infrequent") is learned by `OneHotEncoder(max_categories=11)` inside the pipeline, so it comes from training folds only. Earlier versions picked the top 10 from the full dataset before splitting.

## 5. Limitations

- **One week of 2015 data**, with uneven airline coverage. The results describe this sample; they do not establish trends or seasonality.
- **Label noise.** Model accuracy is about 88% on tweets where annotators fully agreed and much lower on ambiguous ones. This caps what any text model can achieve against these labels.
- **Observational analysis.** Associations between airline and sentiment are not causal.
- **Bag-of-words models** cannot read sarcasm or context beyond short n-grams, and the predicted probabilities are not calibrated.
- **Twitter is a complaint-heavy channel.** Negative shares describe this channel, not overall customer satisfaction.
