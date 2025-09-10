import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from src.feature_extraction import extract_features
from src.data_preprocessing import load_and_preprocess

def evaluate():
    data = load_and_preprocess() if not os.path.exists('data/processed/processed_data.csv') else pd.read_csv('data/processed/processed_data.csv')
    X = data['text']
    y = data['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Use absolute paths for model and vectorizer
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, 'models')
    model = joblib.load(os.path.join(models_dir, 'saved_model.pkl'))
    vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
    X_test_vec = extract_features(X_test, vectorizer, fit=False)
    predictions = model.predict(X_test_vec)
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/confusion_matrix.png')

if __name__ == '__main__':
    evaluate()