# Fake News Detection

## Overview
This project uses NLP techniques to classify news articles as real or fake. It employs text preprocessing, TF-IDF feature extraction, and Logistic Regression for classification.

## Folder Structure
- `data/raw/`: Place downloaded CSVs here (fake.csv, true.csv).
- `data/processed/`: Generated cleaned data.
- `notebooks/`: Jupyter notebooks for exploration, preprocessing, training, and evaluation.
- `src/`: Python scripts for preprocessing, feature extraction, training, evaluation, and prediction.
- `models/`: Saved models and vectorizers.
- `tests/`: Unit tests.
- `reports/`: Figures and final report.

## Setup
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) and place in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: 01 -> 02 -> 03 -> 04.
4. Or use scripts: `python src/train.py` to train and save model.

## Usage
- Train: `python src/train.py`
- Evaluate: `python src/evaluate.py`
- Predict: `python src/predict.py "Your news text here"`
- Tests: `pytest`

## Model
- Preprocessing: Cleaning, stopword removal, lemmatization.
- Features: TF-IDF.
- Classifier: Logistic Regression (accurate for this task).

<!-- ## Results -->
<!-- Accuracy ~95% on test set (run evaluation for exact metrics). -->