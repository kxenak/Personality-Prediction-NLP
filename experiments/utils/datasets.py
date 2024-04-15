import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

class MBTI_Dataset:
    def __init__(self, data_folder, fold_index, is_train=True, is_test=False):
        fold_files = [f for f in os.listdir(os.path.join(data_folder, 'folds')) if f.endswith('.csv')]

        val_fold_file = f'fold_{fold_index}.csv'

        if is_train:
            data_files = [f for f in fold_files if f != val_fold_file]
        else:
            data_files = [val_fold_file]

        # Read and concatenate data
        if is_test:
            self.data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        else:
            self.data = pd.concat([pd.read_csv(os.path.join(data_folder, 'folds', f)) for f in data_files])
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]
        label = self.encode_label(self.data.iloc[idx, 0])
        return text, label

    def encode_label(self, label):
        encoding_dict = {
            'ISTJ': 0, 'ISFJ': 1, 'INFJ': 2, 'INTJ': 3,
            'ISTP': 4, 'ISFP': 5, 'INFP': 6, 'INTP': 7,
            'ESTP': 8, 'ESFP': 9, 'ENFP': 10, 'ENTP': 11,
            'ESTJ': 12, 'ESFJ': 13, 'ENFJ': 14, 'ENTJ': 15
        }
        return encoding_dict[label]

    def create_dataset(self, batch_size=32, shuffle=True):
        text_tensor = tf.ragged.constant([str(t) for t in self.data.iloc[:, 1]])
        label_tensor = tf.ragged.constant([self.encode_label(l) for l in self.data.iloc[:, 0]])

        # One-hot encode the labels
        label_tensor = tf.convert_to_tensor(label_tensor)  # Convert to dense tensor
        label_tensor = tf.one_hot(label_tensor, depth=16)  # Assuming 16 classes

        dataset = Dataset.from_tensor_slices((text_tensor, label_tensor))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))
        dataset = dataset.batch(batch_size)
        return dataset