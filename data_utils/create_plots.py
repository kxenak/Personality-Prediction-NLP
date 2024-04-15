import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_metrics(metrics_folder, plots_folder):
    metric_keys = ['loss', 'binary_accuracy', 'recall', 'precision', 'f1_score']
    num_folds = 5
    
    all_metrics = {key: [] for key in metric_keys}
    
    for i in range(num_folds):
        fold_path = os.path.join(metrics_folder, str(i), 'history.json')
        with open(fold_path, 'r') as file:
            data = json.load(file)
            for key in metric_keys:
                if key == 'f1_score':
                    averaged_f1 = [np.mean(epoch) for epoch in data[key]]
                    all_metrics[key].append(averaged_f1)
                else:
                    all_metrics[key].append(data[key])

    os.makedirs(plots_folder, exist_ok=True)

    plt.style.use('ggplot')

    for key in metric_keys:
        plt.figure(figsize=(10, 5))
        for fold in range(num_folds):
            plt.plot(all_metrics[key][fold], label=f'Fold {fold}')
        plt.title(f'{key} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plots_folder, f'{key}.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from 5-fold CV")
    parser.add_argument('output_folder', type=str, help='Path to the output folder containing train_metrics')
    
    args = parser.parse_args()
    
    train_metrics_folder = os.path.join(args.output_folder, 'train_metrics')
    plots_folder = os.path.join(os.path.dirname(args.output_folder), 'plots')
    plot_metrics(train_metrics_folder, plots_folder)

if __name__ == "__main__":
    main()