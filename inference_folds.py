import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an MBTI model")
    parser.add_argument("--output_dir", type=str, default="output_focal", help="Directory for weights & metrics")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir

    for fold_num in range(5):
        command = f"python3 inference.py --fold_num {fold_num} --output_dir {output_dir}"
        subprocess.run(command, shell=True)
