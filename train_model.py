import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os # Make sure this is imported at the very top

# Ensure these imports correctly reference your local modules
from src.preprocess import preprocess_text
from src.vectorize import create_and_fit_vectorizer, save_vectorizer

def train_spam_detector(
    data_path: str = 'data/sms_spam_collection.csv',
    model_save_path: str = 'models/spam_classifier_model.pkl',
    vectorizer_save_path: str = 'models/tfidf_vectorizer.pkl',
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000
):
    print(f"--- Starting Model Training ---")
    print(f"Loading data from: {data_path}")

    # --- NEW DEBUGGING LINES START HERE ---
    current_dir = os.getcwd()
    data_folder_path = os.path.join(current_dir, 'data')
    print(f"Current working directory (CWD): {current_dir}")
    print(f"Looking for data in: {data_folder_path}")
    try:
        if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
            print(f"Contents of '{data_folder_path}': {os.listdir(data_folder_path)}")
        else:
            print(f"Error: 'data' folder not found or is not a directory at {data_folder_path}.")
    except Exception as e:
        print(f"Error checking 'data' folder contents: {e}")
    # --- NEW DEBUGGING LINES END HERE ---

    # 1. Load Data
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        df = df[['v1', 'v2']] # Adjust column names if different in your dataset
        df.columns = ['label', 'message']
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}. Please ensure the file exists at this exact path.")
        print("Given previous checks, this might indicate a pathing issue from Python's perspective despite the file's presence.")
        return
    except KeyError:
        print("Error: Expected columns 'v1' and 'v2' not found or data is malformed in the CSV. Please check your CSV file content and headers.")
        return
    except pd.errors.EmptyDataError: # Specific for empty but existing files
        print(f"Error: The dataset file at {data_path} is empty or contains no data. Please ensure it's populated with content.")
        return
    except Exception as e: # This is the crucial general catch-all
        print(f"An unexpected error occurred while loading the dataset: {type(e).__name__}: {e}")
        print("Please ensure your 'sms_spam_collection.csv' is a valid, readable CSV file with the expected format.")
        return

    print(f"Initial data shape: {df.shape}")
    print("Label distribution:\n", df['label'].value_counts())

    # 2. Preprocess Data
    print("Preprocessing text data...")
    df['message'] = df['message'].apply(preprocess_text)
    print("Text preprocessing complete.")

    # 3. Vectorize Data
    print("Vectorizing text data...")
    X = create_and_fit_vectorizer(df['message'], vectorizer_save_path, max_features)
    y = df['label']
    print(f"Vectorizer created. Shape of vectorized data: {X.shape}")

    # 4. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Data split: Training samples={X_train.shape[0]}, Testing samples={X_test.shape[0]}")

    # 5. Train Model
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 6. Evaluate Model
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # 7. Save Model
    print(f"Saving trained model to: {model_save_path}")
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

if __name__ == '__main__':
    # Ensure 'models' directory exists before saving
    os.makedirs('models', exist_ok=True)
    train_spam_detector()
    print("\nTraining process finished.")