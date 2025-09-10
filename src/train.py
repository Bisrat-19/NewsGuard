import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score
import joblib
from src.data_preprocessing import load_and_preprocess
from src.feature_extraction import extract_features

def train():
    import os
    data = load_and_preprocess() if not os.path.exists('data/processed/processed_data.csv') else pd.read_csv('data/processed/processed_data.csv')
    # Handle missing values in 'text' column
    data = data.dropna(subset=['text'])
    # Alternatively, to fill missing values with empty string, use:
    # data['text'] = data['text'].fillna('')
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_vec, vectorizer = extract_features(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    X_test_vec = extract_features(X_test, vectorizer, fit=False)
    print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train_vec))}")
    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test_vec))}")
    # Use absolute path for models directory (always, regardless of call context)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'saved_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    print(f"Saving model to: {model_path}")
    print(f"Saving vectorizer to: {vectorizer_path}")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == '__main__':
    train()