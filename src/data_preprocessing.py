import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def load_and_preprocess():
    fake = pd.read_csv('../data/raw/fake.csv')
    true = pd.read_csv('../data/raw/true.csv')
    fake['label'] = 0
    true['label'] = 1
    data = pd.concat([fake, true], ignore_index=True)
    data['text'] = data['text'].apply(clean_text)
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/processed_data.csv', index=False)
    return data

if __name__ == '__main__':
    load_and_preprocess()