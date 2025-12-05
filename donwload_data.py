import kagglehub
import os
import shutil

# Set download path to data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")

print("Downloaded to:", path)

# Move to data folder if not already there
target_path = os.path.join(data_folder, "brats-2021-task1")
if path != target_path and not os.path.exists(target_path):
    shutil.move(path, target_path)
    print(f"Moved to: {target_path}")
else:
    print(f"Dataset location: {path}")