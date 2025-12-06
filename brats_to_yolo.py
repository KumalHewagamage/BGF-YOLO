import os
import random
import shutil
import numpy as np
import nibabel as nib
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
# Path to your downloaded BraTS Data (The folder containing BraTS2021_00000, etc.)
INPUT_DATASET_DIR = 'data/brats-2021-task1/BraTS2021_Training_Data'

# Where to save the converted files
OUTPUT_DIR = "data/brats_2021_yolo"

# Modalities to process
MODALITIES = ['t1', 't2', 't1ce', 'flair']

# ================= HELPER FUNCTIONS =================

def normalize_to_255(img_data):
    """Normalize MRI volume to 0-255 range."""
    if np.max(img_data) == 0: return img_data
    img_data = img_data - np.min(img_data)
    img_data = img_data / np.max(img_data)
    return (img_data * 255).astype(np.uint8)

def get_yolo_bbox(mask_slice, class_id=0):
    """Calculate YOLO bounding box from binary mask."""
    coords = np.argwhere(mask_slice > 0)
    if len(coords) == 0: return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask_slice.shape
    
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_structure():
    """Creates the folder structure: converted/t1/train, converted/t1/test, etc."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Clean start
    
    for mod in MODALITIES:
        os.makedirs(os.path.join(OUTPUT_DIR, mod, 'train'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, mod, 'test'), exist_ok=True)

# ================= MAIN PROCESSING =================

def process_dataset():
    create_structure()
    
    # Get all patient folders
    patient_folders = [f for f in glob(os.path.join(INPUT_DATASET_DIR, "*")) if os.path.isdir(f)]
    print(f"Found {len(patient_folders)} patients. Starting conversion...")

    # We will store file paths to split them later
    # Structure: {'t1': [list_of_filenames], 't2': ...}
    generated_files = {mod: [] for mod in MODALITIES}

    for patient_path in patient_folders:
        patient_id = os.path.basename(patient_path)
        
        # 1. Load Segmentation to pick a slice
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        if not os.path.exists(seg_path): continue
            
        try:
            seg_img = nib.load(seg_path).get_fdata()
        except: continue

        # Pick random slice with tumor
        z_indices = np.where(np.any(seg_img > 0, axis=(0,1)))[0]
        if len(z_indices) > 0:
            slice_idx = random.choice(z_indices)
        else:
            slice_idx = seg_img.shape[2] // 2 # Middle slice if healthy

        # Get Label String (Shared across all modalities for this slice)
        # Rotate 90 deg to fix orientation
        mask_slice = np.rot90(seg_img[:, :, slice_idx])
        label_str = get_yolo_bbox(mask_slice)

        # 2. Process each modality
        for mod in MODALITIES:
            nii_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            if not os.path.exists(nii_path): continue
            
            # Load and Process Image
            img_data = nib.load(nii_path).get_fdata()
            img_slice = np.rot90(img_data[:, :, slice_idx])
            img_final = normalize_to_255(img_slice)
            
            # Define File Names
            base_name = f"{patient_id}_s{slice_idx}"
            img_name = f"{base_name}.jpg"
            txt_name = f"{base_name}.txt"
            
            # Save temporarily in the main mod folder (we will move to train/test later)
            save_path = os.path.join(OUTPUT_DIR, mod)
            
            cv2.imwrite(os.path.join(save_path, img_name), img_final)
            
            # Save Label
            txt_path = os.path.join(save_path, txt_name)
            with open(txt_path, 'w') as f:
                if label_str: f.write(label_str)
                # If no tumor, file is empty (correct for YOLO background images)
            
            generated_files[mod].append(base_name)

    print("Image generation complete. Performing 70/30 Split...")

    # ================= SPLITTING =================
    
    for mod in MODALITIES:
        files = generated_files[mod]
        # Split into train and test
        train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
        
        # Helper to move files
        def move_files(file_list, split_type):
            for base in file_list:
                src_img = os.path.join(OUTPUT_DIR, mod, f"{base}.jpg")
                src_txt = os.path.join(OUTPUT_DIR, mod, f"{base}.txt")
                
                dst_dir = os.path.join(OUTPUT_DIR, mod, split_type)
                
                shutil.move(src_img, os.path.join(dst_dir, f"{base}.jpg"))
                shutil.move(src_txt, os.path.join(dst_dir, f"{base}.txt"))

        move_files(train_files, 'train')
        move_files(test_files, 'test')
        
        print(f"[{mod}] Train: {len(train_files)} | Test: {len(test_files)}")

    print(f"\nDone! Data is ready in '{OUTPUT_DIR}'")

# Run it
if __name__ == "__main__":
    process_dataset()