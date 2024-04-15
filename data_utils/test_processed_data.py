import pickle
import os
import argparse

def load_processed_data(file_path):
    with open(file_path, 'rb') as file:
        processed_data = pickle.load(file)
    return processed_data

def print_sample_data(processed_data, num_samples=5):
    print(f"Total processed entries: {len(processed_data)}")
    print("Sample data:")
    for i in range(min(num_samples, len(processed_data))):
        label, text = processed_data[i]
        print(f"Label: {label}")
        print(f"Text: {text}")
        print("---")

def main():
    parser = argparse.ArgumentParser(description='Load and display processed text data.')
    parser.add_argument('-s', '--stemming', action='store_true', help='Load data processed with stemming')
    parser.add_argument('-l', '--lemmatization', action='store_true', help='Load data processed with lemmatization')
    args = parser.parse_args()

    processed_data_dir = 'processed_data'
    # Decide the file based on the command-line arguments
    if args.stemming:
        processed_file = os.path.join(processed_data_dir, 'processed_data_s.pkl')
    elif args.lemmatization:
        processed_file = os.path.join(processed_data_dir, 'processed_data_l.pkl')
    else:
        print("Please specify whether to load stemming or lemmatization processed data.")
        return

    try:
        processed_data = load_processed_data(processed_file)
        print_sample_data(processed_data)
    except FileNotFoundError:
        print(f"Processed data file '{processed_file}' not found.")
    except Exception as e:
        print(f"Error loading processed data: {e}")

if __name__ == '__main__':
    main()