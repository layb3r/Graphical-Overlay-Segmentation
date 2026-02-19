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
    # Find pixels belonging to this instance
    rows, cols = np.where(mask == instance_id)
    
    if len(rows) == 0:
        return None
    
    # Get bounding box coordinates
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    # Get image dimensions
    img_height, img_width = mask.shape
    
    # Calculate YOLO format (normalized coordinates)
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
    
    # Process both train and val splits
    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        
        images_input = input_base / split / 'images'
        masks_input = input_base / split / 'masks'
        
        images_output = output_base / 'images' / split
        labels_output = output_base / 'labels' / split
        
        # Create output directories
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        
        # Get all mask files
        if not masks_input.exists():
            print(f"Warning: {masks_input} does not exist, skipping...")
            continue
            
        mask_files = sorted(list(masks_input.glob('*.png')))
        print(f"Found {len(mask_files)} mask files")
        
        for mask_path in mask_files:
            # Read mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read mask: {mask_path}")
                continue
            
            # Get unique instance IDs (excluding background, assumed to be 0)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]  # Remove background
            
            # Generate label file
            label_filename = mask_path.stem + '.txt'
            label_path = labels_output / label_filename
            
            # Extract bounding boxes for each instance
            bboxes = []
            for instance_id in instance_ids:
                bbox = get_bounding_box(mask, instance_id)
                if bbox is not None:
                    bboxes.append((class_id, *bbox))
            
            # Write to label file
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    # Format: class_id x_center y_center width height
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            
            # Copy corresponding image
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

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "."  # Directory containing train/ and val/ folders
    OUTPUT_DIR = "yolo_dataset"  # Output directory for YOLO format
    CLASS_ID = 0  # Class ID for all detections
    
    # You can modify these paths as needed
    # INPUT_DIR = "path/to/your/input/data"
    # OUTPUT_DIR = "path/to/your/output/data"
    
    print("="*60)
    print("Instance Mask to YOLO Detection Format Converter")
    print("="*60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Class ID: {CLASS_ID}")
    
    process_masks_to_yolo(INPUT_DIR, OUTPUT_DIR, CLASS_ID)
