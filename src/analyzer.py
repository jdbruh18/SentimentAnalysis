"""High-level intelligence analysis API."""

from __future__ import annotations

from src.model import ThreatClassifier


class Analyzer:
    """Facade for training and querying the threat-signal classifier."""

    def __init__(self) -> None:
        self._classifier = ThreatClassifier()

    def train(self, texts: list[str], labels: list[int]) -> float:
        """Train the underlying classifier.

        Parameters
        ----------
        texts:
            Raw text samples.
        labels:
            Numeric labels (0 = negative/benign, 4 = positive/threat).

        Returns
        -------
        float
            Accuracy score on the held-out test split.
        """
        return self._classifier.fit(texts, labels)

    def analyze(self, texts: list[str]) -> list[dict]:
        """Return analysis results for a list of intelligence signals.

        Each result is a dict with keys ``text`` and ``classification``.
        """
        classifications = self._classifier.predict(texts)
        return [
            {"text": t, "classification": c}
            for t, c in zip(texts, classifications)
        ]
