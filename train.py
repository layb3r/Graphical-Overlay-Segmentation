import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    TrainingArguments,
    Trainer
)
from pycocotools.coco import COCO
from PIL import Image
import cv2
from pathlib import Path
import albumentations as A
from typing import Dict, List
import evaluate

# ============= Dataset Class =============

class COCOInstanceSegmentationDataset(Dataset):
    def __init__(
        self, 
        images_folder: str, 
        annotation_file: str, 
        processor: Mask2FormerImageProcessor,
        augment: bool = False
    ):
        """
        Args:
            images_folder: Path to images folder
            annotation_file: Path to COCO JSON file
            processor: Mask2Former image processor
            augment: Whether to apply data augmentation
        """
        self.images_folder = Path(images_folder)
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.augment = augment
        
        # Get all image IDs
        self.image_ids = self.coco.getImgIds()
        
        # Data augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.images_folder / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare segmentation maps and class labels
        masks = []
        class_labels = []
        
        for ann in anns:
            # Get mask from COCO segmentation
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            class_labels.append(ann['category_id'])
        
        # Apply augmentation if enabled
        if self.transform is not None and len(masks) > 0:
            # Stack masks for augmentation
            masks_np = np.stack(masks, axis=-1) if len(masks) > 0 else np.zeros((image_np.shape[0], image_np.shape[1], 0))
            
            transformed = self.transform(
                image=image_np,
                masks=[masks_np[:, :, i] for i in range(masks_np.shape[2])]
            )
            
            image_np = transformed['image']
            masks = transformed['masks']
        
        # Convert back to PIL for processor
        image = Image.fromarray(image_np)
        
        # Prepare instance segmentation map
        # Each instance gets a unique ID
        instance_seg_map = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.int32)
        
        for i, mask in enumerate(masks):
            instance_seg_map[mask > 0] = i + 1  # Instance IDs start from 1
        
        # Prepare class labels map (same class for all instances in your case)
        class_labels_map = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.int32)
        
        for i, (mask, class_id) in enumerate(zip(masks, class_labels)):
            class_labels_map[mask > 0] = class_id
        
        # Process with Mask2Former processor
        # The processor expects instance and class segmentation maps
        inputs = self.processor(
            images=image,
            segmentation_maps=instance_seg_map,
            instance_id_to_semantic_id={i + 1: class_labels[i] for i in range(len(class_labels))},
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs


# ============= Collate Function =============

def collate_fn(batch):
    """Custom collate function for batching"""
    # Get all keys from first item
    keys = batch[0].keys()
    
    # Initialize output dict
    output = {}
    
    for key in keys:
        if key == 'pixel_values':
            # Stack images
            output[key] = torch.stack([item[key] for item in batch])
        elif key == 'pixel_mask':
            # Stack pixel masks
            output[key] = torch.stack([item[key] for item in batch])
        elif key in ['mask_labels', 'class_labels']:
            # These are lists of tensors, keep as list
            output[key] = [item[key] for item in batch]
        else:
            # Default stacking
            try:
                output[key] = torch.stack([item[key] for item in batch])
            except:
                output[key] = [item[key] for item in batch]
    
    return output


# ============= Evaluation Metric =============

class SegmentationMetrics:
    def __init__(self):
        self.metric = evaluate.load("mean_iou")
    
    def compute_metrics(self, eval_pred):
        """Compute mIoU and other metrics"""
        logits, labels = eval_pred
        
        # This is a simplified version - you may need to adapt based on your needs
        predictions = logits.argmax(axis=1)
        
        metrics = self.metric.compute(
            predictions=predictions,
            references=labels,
            num_labels=2,  # Adjust based on your number of classes
            ignore_index=255
        )
        
        return {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"]
        }


# ============= Training Setup =============

def setup_model_and_processor(num_classes=1, pretrained_model="facebook/mask2former-swin-tiny-coco-instance"):
    """
    Setup Mask2Former model and processor
    
    Args:
        num_classes: Number of classes (1 for single class instance segmentation)
        pretrained_model: HuggingFace model ID
    """
    # Load processor
    processor = Mask2FormerImageProcessor.from_pretrained(pretrained_model)
    
    # Load model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    return model, processor


def train_mask2former(
    train_images_folder: str,
    train_annotation_file: str,
    val_images_folder: str,
    val_annotation_file: str,
    output_dir: str = "./mask2former-finetuned",
    num_classes: int = 1,
    pretrained_model: str = "facebook/mask2former-swin-tiny-coco-instance",
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    eval_steps: int = 100,
    save_steps: int = 100,
):
    """
    Fine-tune Mask2Former on custom dataset
    """
    
    # Setup model and processor
    print("Loading model and processor...")
    model, processor = setup_model_and_processor(num_classes, pretrained_model)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = COCOInstanceSegmentationDataset(
        images_folder=train_images_folder,
        annotation_file=train_annotation_file,
        processor=processor,
        augment=True
    )
    
    val_dataset = COCOInstanceSegmentationDataset(
        images_folder=val_images_folder,
        annotation_file=val_annotation_file,
        processor=processor,
        augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_model")
    
    print(f"Training complete! Model saved to {output_dir}/final_model")
    
    return model, processor, trainer


# ============= Main Training Script =============

if __name__ == "__main__":
    # Configuration
    TRAIN_IMAGES = "dataset/images/train"
    TRAIN_ANNOTATIONS = "dataset/coco_annotations/train.json"
    VAL_IMAGES = "dataset/images/val"
    VAL_ANNOTATIONS = "dataset/coco_annotations/val.json"
    
    OUTPUT_DIR = "./mask2former-overlay-segmentation"
    NUM_CLASSES = 1  # Single class: 
    
    # # Smaller, faster (recommended for starting)
    # PRETRAINED_MODEL = "facebook/mask2former-swin-tiny-coco-instance"

    # # Medium size
    # PRETRAINED_MODEL = "facebook/mask2former-swin-small-coco-instance"

    # # Larger, more accurate but slower
    # PRETRAINED_MODEL = "facebook/mask2former-swin-base-IN21k-coco-instance"


    PRETRAINED_MODEL = "facebook/mask2former-swin-small-coco-instance"  # Options: swin-tiny, swin-small, swin-base-IN21k
    
    # Training hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 4  # Reduce if you run out of memory
    LEARNING_RATE = 1e-5
    
    # Train the model
    model, processor, trainer = train_mask2former(
        train_images_folder=TRAIN_IMAGES,
        train_annotation_file=TRAIN_ANNOTATIONS,
        val_images_folder=VAL_IMAGES,
        val_annotation_file=VAL_ANNOTATIONS,
        output_dir=OUTPUT_DIR,
        num_classes=NUM_CLASSES,
        pretrained_model=PRETRAINED_MODEL,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )