"""Text preprocessing utilities for intelligence signal analysis."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lower-case, remove URLs, mentions, punctuation, and digits."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens."""
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop common English stop words."""
    return [t for t in tokens if t not in _stop_words]


def stem(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to each token."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str) -> str:
    """Full preprocessing pipeline: clean → tokenise → remove stopwords → stem → rejoin."""
    tokens = stem(remove_stopwords(tokenize(clean_text(text))))
    return " ".join(tokens)
