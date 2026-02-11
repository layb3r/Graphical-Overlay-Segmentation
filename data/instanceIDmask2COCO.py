import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from pycocotools import mask as mask_util
from PIL import Image

class InstanceIDMaskToCOCO:
    def __init__(self, class_name="overlay_element"):
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
        binary_mask = (binary_mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 1 or len(contour) < 3:
                continue
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
        return polygons
    
    def polygon_to_bbox(self, polygon):
        """Same as above"""
        x_coords = polygon[0::2]
        y_coords = polygon[1::2]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    
    def process_instance_mask(self, mask_path, image_id):
        """Process instance ID mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            return 0
        
        # Get unique instance IDs (excluding background=0)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        annotations_added = 0
        
        for instance_id in instance_ids:
            # Create binary mask for this instance
            instance_mask = (mask == instance_id).astype(np.uint8) * 255
            
            # Convert to polygons
            polygons = self.binary_mask_to_polygon(instance_mask)
            
            for polygon in polygons:
                bbox = self.polygon_to_bbox(polygon)
                area = float(np.sum(mask == instance_id))
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
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
        """Process dataset with instance ID masks"""
        images_folder = Path(images_folder)
        masks_folder = Path(masks_folder)
        
        image_files = sorted(list(images_folder.glob("*.png")) + 
                           list(images_folder.glob("*.jpg")))
        
        total_instances = 0
        
        for image_path in image_files:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Add image
            self.coco_format["images"].append({
                "id": self.image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            })
            
            current_image_id = self.image_id
            self.image_id += 1
            
            # Find corresponding mask
            mask_path = masks_folder / image_path.name
            
            if not mask_path.exists():
                # Try with _mask suffix
                mask_path = masks_folder / f"{image_path.stem}_mask.png"
            
            if mask_path.exists():
                instances = self.process_instance_mask(mask_path, current_image_id)
                total_instances += instances
                print(f"  {image_path.name}: {instances} instances")
            else:
                print(f"Warning: No mask for {image_path.name}")
        
        print(f"\nTotal: {self.image_id - 1} images, {total_instances} instances")
    
    def save(self, output_path):
        """Same as above"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.coco_format, f, indent=2)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Usage for instance ID masks
    # converter = InstanceIDMaskToCOCO(class_name="overlay_element")
    # converter.process_dataset("dataset/images/train", "dataset/masks/train")
    # converter.save("dataset/coco_annotations/train.json")

    # For training set
    print("Converting training set...")
    converter_train = InstanceIDMaskToCOCO(class_name="overlay_element")
    converter_train.process_dataset(
        images_folder="dataset/images/train",
        masks_folder="dataset/masks/train"
    )
    converter_train.save("dataset/coco_annotations/train.json")
    
    print("\n" + "="*50 + "\n")
    
    # For validation set
    print("Converting validation set...")
    converter_val = InstanceIDMaskToCOCO(class_name="overlay_element")
    converter_val.process_dataset(
        images_folder="dataset/images/val",
        masks_folder="dataset/masks/val"
    )
    converter_val.save("dataset/coco_annotations/val.json")