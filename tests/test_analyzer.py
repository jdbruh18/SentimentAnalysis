"""Unit tests for the defence-intelligence-ai modules."""

import pytest

from src.preprocessor import clean_text, preprocess, remove_stopwords, stem, tokenize
from src.analyzer import Analyzer


# ---------------------------------------------------------------------------
# preprocessor tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO") == "hello"

    def test_removes_urls(self):
        assert "http" not in clean_text("Visit http://example.com now")

    def test_removes_mentions(self):
        assert "@user" not in clean_text("Hello @user!")

    def test_removes_punctuation(self):
        result = clean_text("Hello, world!")
        assert "," not in result and "!" not in result


class TestTokenize:
    def test_splits_on_whitespace(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["this", "is", "a", "threat"]
        result = remove_stopwords(tokens)
        assert "threat" in result
        assert "this" not in result

    def test_empty_list(self):
        assert remove_stopwords([]) == []


class TestStem:
    def test_stems_word(self):
        result = stem(["running", "jumps"])
        assert result == ["run", "jump"]

    def test_empty_list(self):
        assert stem([]) == []


class TestPreprocess:
    def test_returns_string(self):
        assert isinstance(preprocess("Threat detected at 0600"), str)

    def test_pipeline(self):
        result = preprocess("Running towards the border quickly")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# analyzer tests
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "Suspicious movement near the perimeter",
    "All clear, no incidents reported",
    "Unidentified object detected",
    "Routine maintenance completed",
] * 20

TRAIN_LABELS = [4, 0, 4, 0] * 20


class TestAnalyzer:
    @pytest.fixture
    def trained_analyzer(self):
        a = Analyzer()
        a.train(TRAIN_TEXTS, TRAIN_LABELS)
        return a

    def test_train_returns_accuracy(self):
        a = Analyzer()
        acc = a.train(TRAIN_TEXTS, TRAIN_LABELS)
        assert 0.0 <= acc <= 1.0

    def test_analyze_returns_list_of_dicts(self, trained_analyzer):
        results = trained_analyzer.analyze(["Suspicious activity detected"])
        assert isinstance(results, list)
        assert len(results) == 1
        assert "text" in results[0]
        assert "classification" in results[0]

    def test_analyze_classification_values(self, trained_analyzer):
        results = trained_analyzer.analyze(["test signal"])
        assert results[0]["classification"] in {"positive", "negative"}

    def test_analyze_before_train_raises(self):
        a = Analyzer()
        with pytest.raises(RuntimeError):
            a.analyze(["some text"])
