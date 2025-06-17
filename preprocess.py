import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    """
    Downloads necessary NLTK data if not already present.
    nltk.download() itself checks if data is already installed.
    """
    print("Ensuring NLTK 'stopwords' are downloaded...")
    nltk.download('stopwords', quiet=True) # quiet=True suppresses verbose output if already downloaded
    print("Ensuring NLTK 'wordnet' are downloaded...")
    nltk.download('wordnet', quiet=True)

# Call the download function when the module is loaded
download_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    # ... (rest of your preprocess_text function)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

if __name__ == '__main__':
    # ... (rest of your __main__ block)
    sample_message = "Free entry in a 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    processed = preprocess_text(sample_message)
    print(f"Original: {sample_message}")
    print(f"Processed: {processed}")