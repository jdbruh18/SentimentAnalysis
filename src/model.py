"""Logistic-regression model for threat-signal classification."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessor import preprocess

LABELS = {0: "negative", 4: "positive"}


class ThreatClassifier:
    """Wrapper around a TF-IDF + Logistic Regression pipeline."""

    def __init__(self, max_features: int = 50_000, test_size: float = 0.2) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.test_size = test_size
        self._trained = False

    def fit(self, texts: list[str], labels: list[int]) -> float:
        """Preprocess, vectorise, train, and return accuracy on the held-out split."""
        processed = [preprocess(t) for t in texts]
        X = self.vectorizer.fit_transform(processed)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        self.clf.fit(X_train, y_train)
        self._trained = True

        y_pred = self.clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self._plot_confusion_matrix(y_test, y_pred)
        return acc

    def predict(self, texts: list[str]) -> list[str]:
        """Return human-readable labels for a list of raw texts."""
        if not self._trained:
            raise RuntimeError("Model must be trained before calling predict().")
        processed = [preprocess(t) for t in texts]
        X = self.vectorizer.transform(processed)
        raw = self.clf.predict(X)
        return [LABELS.get(int(label), str(label)) for label in raw]

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        os.makedirs("results", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Threat Classifier – Confusion Matrix")
        plt.tight_layout()
        plt.savefig("results/confusion_matrix.png")
        plt.close()
