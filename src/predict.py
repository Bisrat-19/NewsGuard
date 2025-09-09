import sys
import joblib
from src.data_preprocessing import clean_text
from src.feature_extraction import extract_features

def predict(text):
    model = joblib.load('models/saved_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    cleaned = clean_text(text)
    vec = extract_features([cleaned], vectorizer, fit=False)
    pred = model.predict(vec)[0]
    return 'Real' if pred == 1 else 'Fake'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        print(f"Prediction: {predict(text)}")
    else:
        print("Usage: python src/predict.py \"Your news text here\"")