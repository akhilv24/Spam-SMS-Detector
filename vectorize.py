import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def create_and_fit_vectorizer(messages, vectorizer_path, max_features=None):
    """
    Creates and fits a TF-IDF Vectorizer to the messages, then saves it.
    It accepts messages, a path to save the vectorizer, and an optional max_features argument.
    """
    if max_features:
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(messages)
    save_vectorizer(vectorizer, vectorizer_path) 
    return X

def save_vectorizer(vectorizer, path):
    """Saves the trained TF-IDF vectorizer to the specified path."""
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {path}")

def load_vectorizer(path):
    """Loads a saved TF-IDF vectorizer from the specified path."""
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

if __name__ == '__main__':
    # This block is for testing vectorize.py directly
    sample_texts = [
        "this is a sample message about free stuff",
        "another sample message to test vectorization",
        "free offer available now",
        "just a normal chat message"
    ]
    test_vectorizer_path = "../models/test_tfidf_vectorizer.pkl"
    # Ensure 'models' directory exists for testing
    import os
    os.makedirs(os.path.dirname(test_vectorizer_path), exist_ok=True)

    X_vectorized = create_and_fit_vectorizer(sample_texts, test_vectorizer_path, max_features=10)
    print("Vectorized data shape (from test run):", X_vectorized.shape)

    # Example of loading the saved vectorizer for test
    loaded_vec = load_vectorizer(test_vectorizer_path)
    print(f"Loaded vectorizer type: {type(loaded_vec)}")