import pickle
import os
from src.preprocess import preprocess_text
from src.vectorize import load_vectorizer
def load_model(filepath: str):
    """
    Loads a saved machine learning model from a file.

    Args:
        filepath (str): The path to the saved model (.pkl file).

    Returns:
        object: The loaded machine learning model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}. Please train the model first.")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_message_spam(
    message: str,
    model_path: str = 'models/spam_classifier_model.pkl',
    vectorizer_path: str = 'models/tfidf_vectorizer.pkl'
) -> str:
    """
    Predicts whether a given message is 'Spam' or 'Ham'.

    Args:
        message (str): The raw input message to classify.
        model_path (str): Path to the saved trained model.
        vectorizer_path (str): Path to the saved TF-IDF vectorizer.

    Returns:
        str: 'Spam' or 'Ham'.
    """
    try:
        # Load the model and vectorizer
        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run 'train_model.py' first to save the model and vectorizer.")
        return "Prediction Error (Model/Vectorizer not found)"

    # Preprocess the input message
    processed_message = preprocess_text(message)

    # Vectorize the preprocessed message
    # Use transform, not fit_transform, as the vectorizer is already fitted
    vectorized_message = vectorizer.transform([processed_message])

    # Make prediction
    prediction = model.predict(vectorized_message)[0]

    return "Spam" if prediction == 1 else "Ham"

if __name__ == '__main__':
    print("--- Spam Message Predictor ---")
    print("Note: Ensure 'train_model.py' has been run to save the model and vectorizer.")

    while True:
        user_input = input("\nEnter a message to classify (or 'quit' to exit): \n")
        if user_input.lower() == 'quit':
            break

        result = predict_message_spam(user_input)
        print(f"The message is classified as: **{result}**")

    print("\n--- Exiting Predictor ---")