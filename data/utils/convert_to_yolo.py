import os
import numpy as np
import cv2
from pathlib import Path
import shutil

def get_bounding_box(mask, instance_id):
    """
    Extract bounding box for a specific instance in the mask.
    Returns normalized [x_center, y_center, width, height] for YOLO format.
    """
    rows, cols = np.where(mask == instance_id)
    
    if len(rows) == 0:
        return None
    
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    img_height, img_width = mask.shape
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min + 1) / img_width
    height = (y_max - y_min + 1) / img_height
    
    return x_center, y_center, width, height

def process_masks_to_yolo(input_base_dir, output_base_dir, class_id=0):
    """
    Convert instance masks to YOLO detection format.
    
    Args:
        input_base_dir: Base directory containing train/ and val/ folders
        output_base_dir: Output directory for YOLO format dataset
        class_id: Class ID to use for all detections (default: 0)
    """
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        images_input = input_base / split / 'images'
        masks_input = input_base / split / 'masks'
        
        images_output = output_base / 'images' / split
        labels_output = output_base / 'labels' / split
        
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        
        if not masks_input.exists():
            print(f"Warning: {masks_input} does not exist, skipping...")
            continue
            
        mask_files = sorted(list(masks_input.glob('*.png')))
        print(f"Found {len(mask_files)} mask files")
        
        for mask_path in mask_files:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read mask: {mask_path}")
                continue
            
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]  # Remove background
            
            label_filename = mask_path.stem + '.txt'
            label_path = labels_output / label_filename
            
            bboxes = []
            for instance_id in instance_ids:
                bbox = get_bounding_box(mask, instance_id)
                if bbox is not None:
                    bboxes.append((class_id, *bbox))
            
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # Format: class_id x_center y_center width height
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            
            image_filename = mask_path.name
            image_input_path = images_input / image_filename
            image_output_path = images_output / image_filename
            
            if image_input_path.exists():
                shutil.copy2(image_input_path, image_output_path)
            else:
                print(f"Warning: Image not found: {image_input_path}")
            
            print(f"Processed {mask_path.name}: {len(bboxes)} instances")
    
    print("\nConversion complete!")
    print(f"Output directory: {output_base}")