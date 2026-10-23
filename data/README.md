# Data

The dataset is **not included** in this repository; see the main README's *Dataset* section for the source and licence.

```
data/
├── raw/Tweets.csv                   # download from Kaggle (14,640 rows × 15 columns)
└── processed/cleaned_tweets.csv     # created by notebooks/02_data_cleaning.ipynb (14,481 rows × 23 columns)
```

Both folders are git-ignored.

## Processed columns

| Column | Description |
|---|---|
| `tweet_id` | Tweet identifier (string) |
| `airline_sentiment` | Human label: Positive / Neutral / Negative |
| `airline_sentiment_confidence` | Annotator agreement for the label (0–1) |
| `negativereason` | Complaint category; `Not Applicable` for non-negative tweets |
| `negativereason_confidence` | Annotator agreement for the category; 0 when not applicable |
| `airline` | Airline, after the documented Delta → JetBlue correction |
| `airline_source` | Airline label exactly as in the source file |
| `retweet_count` | Retweets at collection time |
| `text` | Tweet text (contains @usernames; notebooks mask them when displaying tweets) |
| `tweet_created` | Timestamp (UTC) |
| `tweet_location`, `user_timezone` | Optional profile fields; `Unknown` when blank |
| `tweet_date`, `year`, `month`, `month_number`, `day`, `day_name`, `hour` | Calendar features in the source's UTC−08:00 clock time |
| `sentiment_score` | Positive = 1, Neutral = 0, Negative = −1 |
| `is_positive`, `is_neutral`, `is_negative` | 1/0 sentiment flags |

The source's `name` (username), `tweet_coord`, `airline_sentiment_gold` and `negativereason_gold` columns are dropped.
