import subprocess
from sklearn.model_selection import KFold
import os

# Function to train a single fold
def train_fold(train_data, valid_data, fold_num, script_path):
    command = f"python3 {script_path} --train_data={train_data} --valid_data={valid_data} --fold={fold_num}"
    subprocess.run(command, shell=True)

def train_folds(data_path, script_path, num_folds=3):
    # Read your data and extract paths for each fold (assuming foldX.csv files)
    fold_paths_tain = [f"{data_path}/fold_{fold}_train.csv" for fold in range(1, num_folds + 1)]
    fold_paths_valid = [f"{data_path}/fold_{fold}_valid.csv" for fold in range(1, num_folds + 1)]

    for fold, (fold_train, fold_valid) in enumerate(zip(fold_paths_tain, fold_paths_valid)):
        train_data = fold_train  # Assuming a single training file for simplicity
        valid_data = fold_valid # Assuming a single validation file for simplicity

        print(f"Training Fold {fold + 1}")
        # command = f"python3 {os.path(script_path)} --train_data={os.path(train_data)} --valid_data={os.path(valid_data)} --fold={fold + 1}"
        # subprocess.run(command, shell=True)
        train_fold(train_data, valid_data, fold + 1, script_path)

if __name__ == "__main__":
    data_path = "/home/KutumLabGPU/Documents/oralcancer/oc_binary_classifier/data-osmf-norm/folds"  # Change this to the actual path
    script_path = "/home/KutumLabGPU/Documents/oralcancer/oc_binary_classifier/train.py"
    train_folds(data_path, script_path)
