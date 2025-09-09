import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from src.data_preprocessing import load_and_preprocess
from src.feature_extraction import extract_features

def train():
    data = load_and_preprocess() if not os.path.exists('data/processed/processed_data.csv') else pd.read_csv('data/processed/processed_data.csv')
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_vec, vectorizer = extract_features(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    X_test_vec = extract_features(X_test, vectorizer, fit=False)
    print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train_vec))}")
    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test_vec))}")
    joblib.dump(model, 'models/saved_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    train()