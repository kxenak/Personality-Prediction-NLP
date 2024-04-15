from utils.datasets import MBTI_Dataset
import tensorflow as tf
import os
import argparse
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_history(history, filename):
    history_dict = history.history
    history_dict['epochs'] = history.epoch

    with open(filename, 'w') as f:
        json.dump(history_dict, f, cls=NumpyEncoder)

def main(args):
    if args.device == 'cpu':
        device = '/cpu:0'
    elif args.device == 'gpu':
        device = '/gpu:0'
    else:
        raise ValueError(f"Invalid device: {args.device}")

    with tf.device(device):
        # Initialize Paths
        data_dir = 'data/'
        fold_index = args.fold

        # Create the datasets
        train_dataset = MBTI_Dataset(data_dir, fold_index, is_train=True).create_dataset(batch_size=args.batch_size)
        test_dataset = MBTI_Dataset(data_dir, fold_index, is_train=False).create_dataset(batch_size=args.batch_size, shuffle=False)

        # Create the text encoder
        VOCAB_SIZE = 10000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(train_dataset.map(lambda text, label: text))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Model
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 128, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.CategoricalFocalCrossentropy(), \
                    metrics=['accuracy', 'Recall', 'Precision', 'F1Score'])

        # Create save directories
        weights_save_path = f"output_sparse/weights/{args.fold}"
        metrics_save_path = f"output_sparse/train_metrics/{args.fold}"
        os.makedirs(weights_save_path, exist_ok=True)
        os.makedirs(metrics_save_path, exist_ok=True)

        # Create checkpoint callbacks
        checkpoint_epoch = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_save_path, 'model'),
            save_weights_only=False,
            save_best_only=False,
            save_freq=args.save_every * len(train_dataset),
            verbose=1,
            save_format='tf'
        )
        checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_save_path, 'best_model'),
            save_weights_only=False,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_format='tf'
        )

        history = model.fit(train_dataset, epochs=args.epochs, validation_data=test_dataset, callbacks=[checkpoint_epoch, checkpoint_best, early_stopping])
        history_filename = os.path.join(metrics_save_path, 'history.json')
        save_history(history, history_filename)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an MBTI model")
    parser.add_argument("--fold", type=int, required=True, help="Fold to train on")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--save_every", type=int, default=5, help="Save model every n epochs")
    parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'], help="Device to use for training")
    args = parser.parse_args()
    main(args)