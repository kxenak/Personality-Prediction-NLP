import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tqdm
import pickle
import os
import argparse

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

parser = argparse.ArgumentParser(description='Preprocess text data from a CSV file.')
parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file')
parser.add_argument('-s', '--stemming', action='store_true', help='Use stemming for word normalization')
parser.add_argument('-l', '--lemmatization', action='store_true', help='Use lemmatization for word normalization')
args = parser.parse_args()

# Ensure the processed_data directory exists
processed_data_dir = 'processed_data'
os.makedirs(processed_data_dir, exist_ok=True)

# Opening the file using latin-1 encoding
with open(args.file, 'r', encoding='latin-1') as file:
    data = file.readlines()

def is_valid_label(label):
    """Check if the label matches the MBTI format."""
    return re.match(r"^[IE][NS][TF][JP]$", label) is not None

def is_valid_word(word):
    """Check if the word contains only English alphabet letters."""
    return word.isalpha()

def preprocess_text(text, use_stemming=False, use_lemmatization=False):
    try:
        text = text.replace('|||', ' ')  # Replace delimiter with space
        text = re.sub(r'http\S+', '', text)  # Remove web links
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        
        stop_words = set(stopwords.words('english'))
        words = text.split()
        # Filter out stopwords and words containing any digits or non-English alphabet characters
        filtered_words = [word for word in words if word not in stop_words and is_valid_word(word)]
        
        if use_stemming:
            stemmer = PorterStemmer()
            processed_words = [stemmer.stem(word) for word in filtered_words]  # Perform stemming
        elif use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            processed_words = [lemmatizer.lemmatize(word) for word in filtered_words]  # Perform lemmatization
        else:
            processed_words = filtered_words
        
        processed_text = ' '.join(processed_words)  # Join words back into a string
        
        return processed_text
    except Exception as e:
        print(f"Error processing text: {e}")
        return ''

if __name__ == '__main__':
    output_file = os.path.join(processed_data_dir, 'processed_data_s.pkl' if args.stemming else 'processed_data_l.pkl')
    processed_data = []
    if os.path.exists(output_file):
        with open(output_file, 'rb') as file:
            processed_data = pickle.load(file)

    remaining_data = data[len(processed_data):]
    batch_size = 1000

    for i in tqdm.tqdm(range(0, len(remaining_data), batch_size), desc='Processing'):
        batch = remaining_data[i:i+batch_size]
        processed_batch = []
        for row in batch:
            try:
                label, text = row.split(',', 1)
                if is_valid_label(label):
                    processed_text = preprocess_text(text, use_stemming=args.stemming, use_lemmatization=args.lemmatization)
                    processed_batch.append((label, processed_text))
                else:
                    print(f"Skipping invalid label row: {label}")
            except ValueError:
                print(f"Skipping malformed row: {row.strip()}")
        processed_data.extend(processed_batch)
        
        if i % (batch_size * 10) == 0:
            with open(output_file, 'wb') as file:
                pickle.dump(processed_data, file)

    with open(output_file, 'wb') as file:
        pickle.dump(processed_data, file)