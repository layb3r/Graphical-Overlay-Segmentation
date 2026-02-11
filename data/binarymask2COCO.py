import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from pycocotools import mask as mask_util
from PIL import Image

# Structure Assumption:
# dataset/
# ├── images/
# │   ├── image_001.png
# │   ├── image_002.png
# │   └── ...
# └── masks/
#     ├── image_001/
#     │   ├── instance_0.png  # Binary mask: 255=object, 0=background
#     │   ├── instance_1.png
#     │   └── instance_2.png
#     ├── image_002/
#     │   ├── instance_0.png
#     │   └── instance_1.png
#     └── ...

class BinaryMaskToCOCO:
    def __init__(self, class_name="overlay_element"):
        """
        Initialize COCO dataset structure
        
        Args:
            class_name: Name of your single class
        """
        self.coco_format = {
            "info": {
                "description": "Graphical Overlay Elements Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [],
            "categories": [
                {
                    "id": 1,
                    "name": class_name,
                    "supercategory": "ui_element"
                }
            ],
            "images": [],
            "annotations": []
        }
        self.image_id = 1
        self.annotation_id = 1
    
    def binary_mask_to_polygon(self, binary_mask):
        """
        Convert binary mask to polygon(s)
        
        Args:
            binary_mask: numpy array (H, W) with 0 and 255 values
            
        Returns:
            list of polygons [[x1,y1,x2,y2,...], ...]
        """
        # Ensure binary
        binary_mask = (binary_mask > 128).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress contours
        )
        
        polygons = []
        for contour in contours:
            # Flatten contour to polygon format
            contour = contour.squeeze()
            
            # Skip invalid contours
            if len(contour.shape) == 1 or len(contour) < 3:
                continue
            
            # Convert to [x1,y1,x2,y2,...] format
            polygon = contour.flatten().tolist()
            
            # Only keep polygons with at least 6 coordinates (3 points)
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons
    
    def polygon_to_bbox(self, polygon):
        """
        Convert polygon to COCO bbox format [x, y, width, height]
        """
        x_coords = polygon[0::2]
        y_coords = polygon[1::2]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        return [
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min)
        ]
    
    def calculate_area(self, binary_mask):
        """Calculate area from binary mask"""
        return float(np.sum(binary_mask > 128))
    
    def process_single_mask(self, mask_path, image_id):
        """
        Process a single binary mask and add annotation
        
        Args:
            mask_path: Path to binary mask image
            image_id: ID of the parent image
        """
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return 0
        
        # Convert to polygons
        polygons = self.binary_mask_to_polygon(mask)
        
        if not polygons:
            print(f"Warning: No valid polygons found in {mask_path}")
            return 0
        
        # For each polygon (in case of multiple connected components)
        annotations_added = 0
        for polygon in polygons:
            # Calculate bbox and area
            bbox = self.polygon_to_bbox(polygon)
            area = self.calculate_area(mask)
            
            # Create annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Single class
                "segmentation": [polygon],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            self.coco_format["annotations"].append(annotation)
            self.annotation_id += 1
            annotations_added += 1
        
        return annotations_added
    
    def process_dataset(self, images_folder, masks_folder):
        """
        Process entire dataset
        
        Args:
            images_folder: Path to folder containing images
            masks_folder: Path to folder containing mask subfolders
        """
        images_folder = Path(images_folder)
        masks_folder = Path(masks_folder)
        
        # Get all image files
        image_files = sorted(list(images_folder.glob("*.png")) + 
                           list(images_folder.glob("*.jpg")) + 
                           list(images_folder.glob("*.jpeg")))
        
        print(f"Found {len(image_files)} images")
        
        total_instances = 0
        
        for image_path in image_files:
            # Read image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            height, width = img.shape[:2]
            
            # Add image to COCO
            image_info = {
                "id": self.image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
            self.coco_format["images"].append(image_info)
            
            current_image_id = self.image_id
            self.image_id += 1
            
            # Find corresponding mask folder
            mask_folder = masks_folder / image_path.stem
            
            if not mask_folder.exists():
                print(f"Warning: No mask folder for {image_path.name}")
                continue
            
            # Process all instance masks
            mask_files = sorted(mask_folder.glob("instance_*.png"))
            
            if not mask_files:
                # Try alternative naming
                mask_files = sorted(mask_folder.glob("*.png"))
            
            instances_count = 0
            for mask_path in mask_files:
                instances_count += self.process_single_mask(mask_path, current_image_id)
            
            total_instances += instances_count
            print(f"  {image_path.name}: {instances_count} instances")
        
        print(f"\nTotal: {self.image_id - 1} images, {total_instances} instances")
    
    def save(self, output_path):
        """Save COCO JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_format, f, indent=2)
        
        print(f"Saved COCO annotations to {output_path}")


# ============= USAGE =============

if __name__ == "__main__":
    # For training set
    print("Converting training set...")
    converter_train = BinaryMaskToCOCO(class_name="overlay_element")
    converter_train.process_dataset(
        images_folder="dataset/images/train",
        masks_folder="dataset/masks/train"
    )
    converter_train.save("dataset/coco_annotations/train.json")
    
    print("\n" + "="*50 + "\n")
    
    # For validation set
    print("Converting validation set...")
    converter_val = BinaryMaskToCOCO(class_name="overlay_element")
    converter_val.process_dataset(
        images_folder="dataset/images/val",
        masks_folder="dataset/masks/val"
    )
    converter_val.save("dataset/coco_annotations/val.json")