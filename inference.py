import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Mask2FormerInference:
    def __init__(self, model_path: str, device: str = None):
        """
        Load trained Mask2Former model for inference
        
        Args:
            model_path: Path to saved model
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {model_path}...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_path)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict(self, image_path: str, threshold: float = 0.5):
        """
        Run inference on an image
        
        Args:
            image_path: Path to image
            threshold: Confidence threshold for predictions
            
        Returns:
            dict with 'masks', 'scores', 'labels'
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],  # (height, width)
            threshold=threshold
        )[0]
        
        return {
            'masks': results['segmentation'].cpu().numpy(),
            'scores': results.get('scores', None),
            'image': np.array(image)
        }
    
    def visualize(self, image_path: str, threshold: float = 0.5, save_path: str = None):
        """
        Visualize predictions
        """
        results = self.predict(image_path, threshold)
        
        image = results['image']
        masks = results['masks']
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Overlay masks
        axes[1].imshow(image)
        
        # Get unique instance IDs
        instance_ids = np.unique(masks)
        instance_ids = instance_ids[instance_ids != 0]  # Remove background
        
        # Color map for instances
        colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_ids)))
        
        for i, instance_id in enumerate(instance_ids):
            mask = (masks == instance_id).astype(np.uint8)
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask == 1] = [*colors[i][:3], 0.6]  # RGBA
            axes[1].imshow(colored_mask)
        
        axes[1].set_title(f"Predictions ({len(instance_ids)} instances)")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return results


# ============= Usage =============

# Initialize inference
inferencer = Mask2FormerInference(
    model_path="./mask2former-overlay-segmentation/final_model"
)

# Run inference on a single image
results = inferencer.visualize(
    image_path="test_image.png",
    threshold=0.5,
    save_path="prediction.png"
)

print(f"Detected {len(np.unique(results['masks'])) - 1} instances")