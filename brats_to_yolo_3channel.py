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
OUTPUT_DIR = "data/brats_2021_yolo_3channel"

# Select 3 modalities for RGB channels (early fusion)
# Common choices: t1ce, flair, t2 (complementary contrast information)
SELECTED_MODALITIES = ['t1ce', 'flair', 't2']  # These will become R, G, B channels

# ================= HELPER FUNCTIONS =================

def normalize_to_255(img_data):
    """Normalize MRI volume to 0-255 range."""
    if np.max(img_data) == 0: 
        return img_data.astype(np.uint8)
    img_data = img_data - np.min(img_data)
    img_data = img_data / np.max(img_data)
    return (img_data * 255).astype(np.uint8)

def get_yolo_bbox(mask_slice, class_id=0):
    """Calculate YOLO bounding box from binary mask."""
    coords = np.argwhere(mask_slice > 0)
    if len(coords) == 0: 
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask_slice.shape
    
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_structure():
    """Creates the folder structure for 3-channel fused images."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # Clean start
    
    os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)

# ================= MAIN PROCESSING =================

def process_dataset():
    create_structure()
    
    # Get all patient folders
    patient_folders = [f for f in glob(os.path.join(INPUT_DATASET_DIR, "*")) if os.path.isdir(f)]
    print(f"Found {len(patient_folders)} patients. Starting 3-channel fusion...")
    print(f"Selected modalities (RGB): {SELECTED_MODALITIES}")

    # Store generated filenames for train/test split
    generated_files = []
    skipped_count = 0

    for patient_path in patient_folders:
        patient_id = os.path.basename(patient_path)
        
        # 1. Load Segmentation to pick a slice
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        if not os.path.exists(seg_path): 
            skipped_count += 1
            continue
            
        try:
            seg_img = nib.load(seg_path).get_fdata()
        except:
            skipped_count += 1
            continue

        # Pick random slice with tumor (or middle slice if no tumor)
        z_indices = np.where(np.any(seg_img > 0, axis=(0, 1)))[0]
        if len(z_indices) > 0:
            slice_idx = random.choice(z_indices)
        else:
            slice_idx = seg_img.shape[2] // 2

        # Get Label String (bounding box from segmentation mask)
        mask_slice = np.rot90(seg_img[:, :, slice_idx])
        label_str = get_yolo_bbox(mask_slice)

        # 2. Load and stack 3 selected modalities as RGB channels
        channel_imgs = []
        missing_modality = False
        
        for mod in SELECTED_MODALITIES:
            nii_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            if not os.path.exists(nii_path):
                missing_modality = True
                break
            
            try:
                # Load modality volume
                img_data = nib.load(nii_path).get_fdata()
                # Extract same slice and rotate to fix orientation
                img_slice = np.rot90(img_data[:, :, slice_idx])
                # Normalize to 0-255
                img_normalized = normalize_to_255(img_slice)
                channel_imgs.append(img_normalized)
            except:
                missing_modality = True
                break
        
        if missing_modality:
            skipped_count += 1
            continue

        # Stack channels to create RGB image (H x W x 3)
        rgb_image = np.stack(channel_imgs, axis=2)  # Shape: (H, W, 3)
        
        # Define File Names
        base_name = f"{patient_id}_s{slice_idx}"
        img_name = f"{base_name}.png"  # Use PNG for lossless compression
        txt_name = f"{base_name}.txt"
        
        # Save temporarily in the main folder (will split later)
        save_path = OUTPUT_DIR
        
        # Convert RGB to BGR for OpenCV and save
        cv2.imwrite(os.path.join(save_path, img_name), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Save Label
        txt_path = os.path.join(save_path, txt_name)
        with open(txt_path, 'w') as f:
            if label_str:
                f.write(label_str)
            # If no tumor, file is empty (correct for YOLO background images)
        
        generated_files.append(base_name)

    print(f"\nImage generation complete.")
    print(f"Generated: {len(generated_files)} 3-channel images")
    print(f"Skipped: {skipped_count} patients (missing data)")
    print("\nPerforming 70/30 Train/Test Split...")

    # ================= SPLITTING =================
    
    if len(generated_files) == 0:
        print("ERROR: No images generated. Check your INPUT_DATASET_DIR path.")
        return
    
    # Split into train and test
    train_files, test_files = train_test_split(generated_files, test_size=0.3, random_state=42)
    
    # Helper to move files
    def move_files(file_list, split_type):
        for base in file_list:
            src_img = os.path.join(OUTPUT_DIR, f"{base}.png")
            src_txt = os.path.join(OUTPUT_DIR, f"{base}.txt")
            
            dst_dir = os.path.join(OUTPUT_DIR, split_type)
            
            if os.path.exists(src_img):
                shutil.move(src_img, os.path.join(dst_dir, f"{base}.png"))
            if os.path.exists(src_txt):
                shutil.move(src_txt, os.path.join(dst_dir, f"{base}.txt"))

    move_files(train_files, 'train')
    move_files(test_files, 'test')
    
    print(f"\nTrain: {len(train_files)} images")
    print(f"Test:  {len(test_files)} images")
    print(f"\n✅ Done! 3-channel fused data ready in '{OUTPUT_DIR}'")
    print(f"\nChannel mapping:")
    print(f"  R (Red):   {SELECTED_MODALITIES[0]}")
    print(f"  G (Green): {SELECTED_MODALITIES[1]}")
    print(f"  B (Blue):  {SELECTED_MODALITIES[2]}")

# Run it
if __name__ == "__main__":
    process_dataset()
