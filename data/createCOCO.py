from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Single class segmentation

def create_COCO(isBinary=False):
    pass

def visualize_coco_annotations(json_path, num_samples=3):
    """Visualize COCO annotations to verify conversion"""
    coco = COCO(json_path)
    
    # Print statistics
    print(f"Images: {len(coco.getImgIds())}")
    print(f"Annotations: {len(coco.getAnnIds())}")
    print(f"Categories: {coco.loadCats(coco.getCatIds())}")
    
    # Visualize samples
    img_ids = coco.getImgIds()[:num_samples]
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_info['file_name']
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Draw annotations
        coco.showAnns(anns, draw_bbox=True)
        
        plt.title(f"{img_info['file_name']} - {len(anns)} instances")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Validate
    visualize_coco_annotations(
        "dataset/coco_annotations/train.json",
        "dataset/images/train",
        num_samples=5
    )