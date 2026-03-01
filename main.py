"""Entry point for the defence-intelligence-ai analyser."""

from __future__ import annotations

import os

from src.analyzer import Analyzer

SAMPLE_TEXTS = [
    "Suspicious movement detected near the border",
    "Routine patrol completed without incident",
    "Unidentified aircraft spotted in restricted airspace",
    "All systems operational, no anomalies reported",
]

# Toy labels: 4 = threat signal, 0 = benign
SAMPLE_LABELS = [4, 0, 4, 0]


def main() -> None:
    os.makedirs("results", exist_ok=True)

    analyzer = Analyzer()
    print("Training classifier on sample data …")
    accuracy = analyzer.train(SAMPLE_TEXTS * 50, SAMPLE_LABELS * 50)
    print(f"Training complete. Held-out accuracy: {accuracy:.2%}\n")

    print("Analysing new signals:")
    results = analyzer.analyze(SAMPLE_TEXTS)
    for r in results:
        print(f"  [{r['classification'].upper():8s}] {r['text']}")


if __name__ == "__main__":
    main()
