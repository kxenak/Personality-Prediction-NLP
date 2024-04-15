import subprocess
import os

def train_folds():
    for i in range(0,5):
        command = f"python3 train_w.py --fold {i} --device gpu"
        print(command)
        subprocess.run(command, shell=True)

if __name__ == '__main__':
    train_folds()