import os
import json
import argparse
import numpy as np
import tensorflow as tf
from utils.datasets import MBTI_Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def decode_label(label):
    decoding_dict = {0: ['I', 'N', 'T', 'J'], 1: ['E', 'S', 'F', 'P']}
    mbti_type = ''
    for i in range(len(label)):
        mbti_type += decoding_dict[label[i]][i]
    return mbti_type

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an MBTI model")
    parser.add_argument("--fold_num", type=int, required=True, help="Fold number for evaluation")
    parser.add_argument("--output_dir", type=str, default="output_focal", help="Output directory for metrics and plots")
    parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'], help="Device to use for evaluation")
    return parser.parse_args()

def main(args):
    fold_num = args.fold_num
    output_dir = args.output_dir
    
    if args.device == 'cpu':
        device = '/cpu:0'
    elif args.device == 'gpu':
        device = '/gpu:0'
    else:
        raise ValueError(f"Invalid device: {args.device}")

    with tf.device(device):
        # Load the trained model
        model_path = f"{output_dir}/weights/{fold_num}/best_model"
        model = tf.keras.models.load_model(model_path)

        # Load the test dataset
        data_dir = 'data/'
        test_dataset = MBTI_Dataset(data_dir, fold_index=fold_num, is_train=False, is_test=True).create_dataset(batch_size=128)

        # Perform inference on the test dataset
        true_labels = []
        predicted_labels = []
        for batch in test_dataset:
            texts, labels = batch
            predicted_batch = model.predict(texts)
            thresholded_output = (predicted_batch >= 0.5).astype(int)
            
            true_labels.extend(labels.numpy())
            predicted_labels.extend(thresholded_output)

        true_labels = np.array(true_labels).flatten()
        predicted_labels = np.array(predicted_labels).flatten()

        # Compute metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Create metrics dir
        metrics_dir = f"{output_dir}/test_metrics/{fold_num}/"
        os.makedirs(metrics_dir, exist_ok=True)

        # Save metrics to JSON file
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

        # Generate confusion matrix plot
        cm = confusion_matrix(true_labels, predicted_labels)
        labels = ['Positive', 'Negative']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'cm.png'))
        plt.close()

        print(f"Metrics and confusion matrix for fold {fold_num} saved successfully in {metrics_dir}.")

if __name__ == '__main__':
    args = parse_args()
    main(args)