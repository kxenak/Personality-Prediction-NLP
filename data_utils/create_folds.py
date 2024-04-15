import argparse
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold, train_test_split

def load_data(file_path):
    """ Load the processed data from a pickle file. """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_splits(data, directory, n_splits=5):
    """ Create K-Folds cross-validation splits and a test set, then save to CSV. """
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    # Convert to DataFrame for easier manipulation and saving
    data_df = pd.DataFrame(data, columns=['label', 'text'])

    # Splitting the data into training (80%) and testing (20%)
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # Save the test data
    test_df.to_csv(os.path.join(directory, 'test.csv'), index=False)

    # Setting up KFold for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Creating distinct folds
    fold_number = 0
    for train_index, val_index in kf.split(train_df):
        fold_df = train_df.iloc[val_index]
        fold_df.to_csv(os.path.join(directory, f'fold_{fold_number}.csv'), index=False)
        fold_number += 1

def main():
    parser = argparse.ArgumentParser(description="Create data splits for cross-validation and testing.")
    parser.add_argument('-s', '--stemming', action='store_true', help='Use data processed with stemming')
    parser.add_argument('-l', '--lemmatization', action='store_true', help='Use data processed with lemmatization')
    args = parser.parse_args()

    processed_data_dir = 'processed_data'
    if args.stemming:
        data_file = os.path.join(processed_data_dir, 'processed_data_s.pkl')
    elif args.lemmatization:
        data_file = os.path.join(processed_data_dir, 'processed_data_l.pkl')
    else:
        print("Please specify -s for stemming or -l for lemmatization.")
        return

    # Load data from the specified pickle file
    data = load_data(data_file)
    
    # Directory to save the split files
    splits_dir = 'data/folds'
    
    # Generate and save the data splits
    save_splits(data, splits_dir)

if __name__ == "__main__":
    main()