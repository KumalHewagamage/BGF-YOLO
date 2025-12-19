import os
import shutil
import numpy as np
import nibabel as nib
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
INPUT_DATASET_DIR = 'data/brats-2021-task1/BraTS2021_Training_Data'
OUTPUT_DIR = "data/brats_yolo_multiclass_v2"

# RGB Channels (Standard: T1ce=Red, T2=Green, FLAIR=Blue)
SELECTED_MODALITIES = ['t1ce', 't2', 'flair'] 

# --- CLASS DEFINITION SETTINGS ---
# OPTION A: Standard Clinical Hierarchy (Use this for SOTA comparison)
# Class 0 (WT): Whole Tumor (1+2+4)
# Class 1 (TC): Tumor Core (1+4) -> often overlaps with ET
# Class 2 (ET): Enhancing Tumor (4)
# CLASS_MAPPING = {
#     0: [1, 2, 4], 
#     1: [1, 4],     
#     2: [4]         
# }

# OPTION B: Visual Distinction (Use this if you want distinct boxes)
# This forces the model to find the "Necrosis" separately from "Enhancing"
# Class 0 (WT): Whole Tumor (1+2+4)
# Class 1 (NC): Necrosis Only (1) -> Small inner box
# Class 2 (ET): Enhancing Only (4) -> Outer ring box
CLASS_MAPPING = {
    0: [1, 2, 4],  # Whole Tumor (Edema + Core)
    1: [1],        # Necrotic Core ONLY (The dark center)
    2: [4]         # Enhancing Tumor ONLY (The bright ring)
}

# ================= HELPER FUNCTIONS =================

def normalize_channel(img_data):
    """Robust Min-Max normalization per channel."""
    min_val = np.nanmin(img_data)
    max_val = np.nanmax(img_data)
    if max_val - min_val == 0:
        return np.zeros_like(img_data, dtype=np.uint8)
    norm_img = (img_data - min_val) / (max_val - min_val)
    return (norm_img * 255).astype(np.uint8)

def get_bounding_box(mask_slice, labels_of_interest):
    """Calculate tight bounding box for specific labels."""
    # Create mask for ONLY the labels we want
    binary_mask = np.isin(mask_slice, labels_of_interest).astype(np.uint8)
    
    coords = np.argwhere(binary_mask > 0)
    if len(coords) == 0: 
        return None
    
    # Calculate box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add 1 pixel padding if possible to catch edge details
    h_img, w_img = mask_slice.shape
    y_min = max(0, y_min - 1)
    x_min = max(0, x_min - 1)
    y_max = min(h_img, y_max + 1)
    x_max = min(w_img, x_max + 1)

    # YOLO format (normalized)
    x_center = ((x_min + x_max) / 2) / w_img
    y_center = ((y_min + y_max) / 2) / h_img
    width = (x_max - x_min) / w_img
    height = (y_max - y_min) / h_img
    
    return x_center, y_center, width, height, (x_min, y_min, x_max, y_max)

def draw_debug_box(img, box_coords, class_id):
    """Draws box on image for verification (Red=WT, Green=Core, Blue=Enhancing)"""
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR format
    x_min, y_min, x_max, y_max = box_coords
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[class_id % 3], 2)
    cv2.putText(img, f"C{class_id}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id % 3], 1)
    return img

def create_structure():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'debug_visuals'), exist_ok=True)

# ================= MAIN PROCESSING =================

def process_dataset():
    create_structure()
    patient_folders = [f for f in glob(os.path.join(INPUT_DATASET_DIR, "*")) if os.path.isdir(f)]
    print(f"Found {len(patient_folders)} patients.")
    
    processed_samples = []
    
    for patient_path in patient_folders:
        patient_id = os.path.basename(patient_path)
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        if not os.path.exists(seg_path): continue
            
        try:
            seg_vol = nib.load(seg_path).get_fdata()
        except: continue

        # --- FIND BEST SLICE (With most distinct classes) ---
        # We prefer a slice that has ALL 3 labels (1, 2, and 4) to verify separation
        # Flatten slices to count labels
        has_necrosis = np.any(seg_vol == 1, axis=(0,1))
        has_enhancing = np.any(seg_vol == 4, axis=(0,1))
        
        # Priority: Slice with 1 AND 4 (to show separation), otherwise just max tumor
        interesting_slices = np.where(has_necrosis & has_enhancing)[0]
        
        if len(interesting_slices) > 0:
            # Pick the one with largest tumor area among interesting slices
            areas = [np.sum(seg_vol[:,:,i] > 0) for i in interesting_slices]
            slice_idx = interesting_slices[np.argmax(areas)]
        else:
            # Fallback to max area
            slice_idx = np.argmax(np.sum(seg_vol > 0, axis=(0,1)))

        # --- LOAD & NORMALIZE CHANNELS ---
        channel_imgs = []
        valid_patient = True
        for mod in SELECTED_MODALITIES:
            nii_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            if not os.path.exists(nii_path):
                valid_patient = False
                break
            vol = nib.load(nii_path).get_fdata()
            # Rotate 90 deg to align with standard vision view
            slice_img = np.rot90(vol[:, :, slice_idx]) 
            channel_imgs.append(normalize_channel(slice_img))
            
        if not valid_patient: continue
        
        # Stack RGB (H, W, 3)
        rgb_image = np.stack(channel_imgs, axis=2)
        debug_img = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR) # Copy for drawing
        
        # --- GENERATE LABELS ---
        mask_slice = np.rot90(seg_vol[:, :, slice_idx])
        label_lines = []
        has_boxes = False
        
        for class_id, labels in CLASS_MAPPING.items():
            result = get_bounding_box(mask_slice, labels)
            if result:
                xc, yc, w, h, pixel_coords = result
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                has_boxes = True
                # Draw on debug image
                debug_img = draw_debug_box(debug_img, pixel_coords, class_id)
        
        if not has_boxes: continue

        # Save info
        base_name = f"{patient_id}_s{slice_idx}"
        processed_samples.append({'name': base_name, 'img': rgb_image, 'lbl': label_lines})

        # Save Debug Image (First 20 only)
        if len(processed_samples) <= 20:
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'debug_visuals', f"{base_name}_debug.jpg"), debug_img)

    # --- SPLIT & SAVE ---
    print(f"Generated {len(processed_samples)} samples. Splitting 80/20...")
    train, val = train_test_split(processed_samples, test_size=0.2, random_state=42)
    
    def save_data(data, split):
        for item in data:
            img_path = os.path.join(OUTPUT_DIR, 'images', split, item['name'] + '.jpg')
            lbl_path = os.path.join(OUTPUT_DIR, 'labels', split, item['name'] + '.txt')
            
            # Save Image (RGB -> BGR for OpenCV)
            cv2.imwrite(img_path, cv2.cvtColor(item['img'], cv2.COLOR_RGB2BGR))
            
            # Save Label
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(item['lbl']))

    save_data(train, 'train')
    save_data(val, 'val')
    print(f"Done! Check '{OUTPUT_DIR}/debug_visuals' to verify your bounding boxes.")

if __name__ == "__main__":
    process_dataset()