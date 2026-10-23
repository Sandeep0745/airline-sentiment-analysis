"""Project-wide constants. Paths are resolved from this file, so they work from any working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Tweets.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "cleaned_tweets.csv"

IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Order used for classification reports and confusion matrices
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
# Order used for descriptive charts (good -> bad)
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
