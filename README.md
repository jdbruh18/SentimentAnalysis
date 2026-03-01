# Defence Intelligence AI

An AI-powered platform for classifying and analysing intelligence signals using Natural Language Processing (NLP) and machine learning.

## Features

- Text preprocessing pipeline (tokenisation, stop-word removal, stemming)
- Logistic-regression threat-signal classifier (75 %+ accuracy baseline)
- High-level `Analyzer` API for rapid integration
- Confusion-matrix visualisation of model performance

## Project Structure

```
defence-intelligence-ai/
├── main.py              # CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── preprocessor.py  # text cleaning utilities
│   ├── model.py         # model training & inference
│   └── analyzer.py      # high-level analysis API
└── tests/
    └── test_analyzer.py
```

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/jdbruh18/SentimentAnalysis.git
   cd SentimentAnalysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analyser:
   ```bash
   python main.py
   ```

## Running Tests

```bash
python -m pytest tests/
```

## Dataset

Uses the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset from Kaggle as the baseline training corpus.

## Future Improvements

- Fine-tune transformer models (BERT / RoBERTa) for higher accuracy.
- Add multi-label threat categorisation.
- Integrate a REST API for real-time signal ingestion.
