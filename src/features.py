"""TF-IDF feature builders. Each call returns a fresh, unfitted transformer."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Word tokens plus single punctuation characters, so "?" and "!" become features
WORD_OR_PUNCTUATION = r"(?u)\b[\w']+\b|[^\w\s]"


def word_tfidf(ngram_range=(1, 2)):
    """Word n-gram TF-IDF used by the baseline models (notebooks 06-07)."""
    return TfidfVectorizer(ngram_range=ngram_range, min_df=2, sublinear_tf=True)


def word_char_tfidf(word_ngram_range=(1, 2)):
    """Word (capped at 40k) + character 3-5-gram (capped at 50k) TF-IDF from notebook 08."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=word_ngram_range, min_df=2, max_features=40_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50_000,
                                 sublinear_tf=True)),
    ])


def final_tfidf(char_ngram_range=(2, 6)):
    """Final feature set (notebook 10): punctuation-aware words, wider character n-grams, no caps."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=WORD_OR_PUNCTUATION)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range, min_df=2, sublinear_tf=True)),
    ])
