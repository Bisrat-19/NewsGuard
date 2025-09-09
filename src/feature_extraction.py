from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(max_features=5000)

def extract_features(texts, vectorizer=None, fit=True):
    if vectorizer is None:
        vectorizer = get_vectorizer()
    if fit:
        return vectorizer.fit_transform(texts), vectorizer
    return vectorizer.transform(texts)