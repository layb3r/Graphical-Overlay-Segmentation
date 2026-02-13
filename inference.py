import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Mask2FormerInference:
    def __init__(self, model_path: str, processor_path: str = None, device: str = None):
        """
        Load trained Mask2Former model for inference
        
        Args:
            model_path: Path to saved model or checkpoint
            processor_path: Path to processor (optional). If None, tries to load from model_path.
                           For checkpoints, use the original pretrained model name or final_model path.
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {model_path}...")
        try:
            # Auto-detect processor path if not provided
            if processor_path is None:
                processor_path = self._find_processor_path(model_path)
            
            print(f"Loading processor from {processor_path}...")
            self.processor = Mask2FormerImageProcessor.from_pretrained(processor_path)
            
            # Load model
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def _find_processor_path(self, model_path: str) -> str:
        """
        Automatically find processor path for checkpoints
        """
        model_path = Path(model_path)
        
        # Check if preprocessor_config.json exists in model_path
        if (model_path / "preprocessor_config.json").exists():
            return str(model_path)
        
        # If it's a checkpoint folder, look for processor in parent directory's final_model
        if "checkpoint" in model_path.name:
            # Try parent/final_model
            final_model_path = model_path.parent / "final_model"
            if (final_model_path / "preprocessor_config.json").exists():
                print(f"Checkpoint detected. Using processor from {final_model_path}")
                return str(final_model_path)
            
            # Try looking in parent directory itself
            if (model_path.parent / "preprocessor_config.json").exists():
                print(f"Checkpoint detected. Using processor from {model_path.parent}")
                return str(model_path.parent)
        
        # Default: try to load from model_path and let it fail with clear error
        return str(model_path)
    
    def predict(self, image_path: str, threshold: float = 0.5, return_raw: bool = False):
        """
        Run inference on an image
        
        Args:
            image_path: Path to image file, PIL Image, or numpy array
            threshold: Confidence threshold for predictions (default 0.5)
                      Lower values (e.g., 0.1-0.3) detect more instances but may have false positives
                      Higher values (e.g., 0.6-0.9) are more conservative
            return_raw: If True, returns raw model outputs for debugging
            
        Returns:
            dict with 'masks', 'scores', 'image', and optionally 'raw_outputs'
        """
        # Handle different input types
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image from {image_path}: {e}")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path.astype(np.uint8)).convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_path)}. Expected str, PIL.Image, or numpy.ndarray")
        
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
        
        # Get unique mask count for debugging
        masks_array = results['segmentation'].cpu().numpy()
        unique_instances = len(np.unique(masks_array)) - 1  # Exclude background
        print(f"Detected {unique_instances} instances with threshold={threshold}")
        
        result_dict = {
            'masks': masks_array,
            'scores': results.get('scores', None),
            'image': np.array(image)
        }
        
        if return_raw:
            result_dict['raw_outputs'] = outputs
            # Print debugging info
            if hasattr(outputs, 'class_queries_logits'):
                print(f"Raw predictions shape: {outputs.class_queries_logits.shape}")
                print(f"Mask predictions shape: {outputs.masks_queries_logits.shape}")
        
        return result_dict
    
    def predict_with_multiple_thresholds(self, image_path: str, thresholds: list = None):
        """
        Try multiple thresholds to find best detection
        
        Args:
            image_path: Path to image
            thresholds: List of thresholds to try (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        """
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        print("\nTrying multiple thresholds:")
        print("=" * 50)
        
        for thresh in thresholds:
            results = self.predict(image_path, threshold=thresh)
            num_instances = len(np.unique(results['masks'])) - 1
            print(f"Threshold {thresh:.1f}: {num_instances} instances")
            if results['scores'] is not None:
                scores_arr = results['scores'].cpu().numpy() if hasattr(results['scores'], 'cpu') else results['scores']
                if len(scores_arr) > 0:
                    print(f"  Scores: {scores_arr}")
        
        print("=" * 50)
        print("\nTip: If all thresholds show 1 instance, your model might need more training.")
        print("Try visualizing with threshold=0.1 to see if there are any weak predictions.\n")
    
    def visualize(self, image_path: str, threshold: float = 0.5, save_path: str = None, show_scores: bool = True):
        """
        Visualize predictions
        
        Args:
            image_path: Path to image
            threshold: Confidence threshold
            save_path: Path to save visualization
            show_scores: Whether to display confidence scores on masks
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
        
        if len(instance_ids) > 0:
            # Color map for instances
            colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_ids)))
            
            # Create a single overlay image with all masks
            overlay = np.zeros((*masks.shape, 4), dtype=np.float32)
            
            for i, instance_id in enumerate(instance_ids):
                mask = (masks == instance_id).astype(bool)
                overlay[mask] = [*colors[i][:3], 0.6]  # RGBA
                
                # Add scores if available and requested
                if show_scores and results['scores'] is not None:
                    # Find center of mask for text placement
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        center_y, center_x = np.mean(y_coords), np.mean(x_coords)
                        score_val = results['scores'][i].item() if hasattr(results['scores'][i], 'item') else results['scores'][i]
                        axes[1].text(center_x, center_y, f"{score_val:.2f}", 
                                   color='white', fontsize=12, weight='bold',
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            # Display the accumulated overlay
            axes[1].imshow(overlay)
        else:
            # No instances detected - add warning
            axes[1].text(0.5, 0.5, 'No instances detected\nTry lowering threshold', 
                       transform=axes[1].transAxes,
                       ha='center', va='center', fontsize=14, color='red',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        axes[1].set_title(f"Predictions ({len(instance_ids)} instances, threshold={threshold})")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return results


# ============= Usage =============

if __name__ == "__main__":
    # Option 1: Load from final_model (has both model and processor)
    inferencer = Mask2FormerInference(
        model_path="./mask2former-overlay-segmentation/final_model"
    )
    
    # Option 2: Load from checkpoint (need to specify processor path separately)
    # inferencer = Mask2FormerInference(
    #     model_path="./mask2former-overlay-segmentation/checkpoint-2000",
    #     processor_path="./mask2former-overlay-segmentation/final_model"  # or the original pretrained model
    # )
    
    # Option 3: Load checkpoint with original pretrained processor
    # inferencer = Mask2FormerInference(
    #     model_path="./mask2former-overlay-segmentation/checkpoint-2000",
    #     processor_path="facebook/mask2former-swin-small-coco-instance"
    # )
    
    # DEBUGGING: If getting only 1 instance, try multiple thresholds
    print("\n=== DIAGNOSTIC: Testing multiple thresholds ===")
    inferencer.predict_with_multiple_thresholds("test_image.png")
    
    # Run inference with lower threshold (try 0.1-0.3 for early checkpoints)
    print("\n=== Visualizing with threshold=0.3 ===")
    results = inferencer.visualize(
        image_path="test_image.png",
        threshold=0.3,  # Lower threshold for early checkpoints
        save_path="prediction.png"
    )
    
    print(f"\nFinal count: {len(np.unique(results['masks'])) - 1} instances")
    print("\nIf you're still getting 1 instance:")
    print("1. Your model needs more training (2000 steps might be too early)")
    print("2. Check training loss - is it decreasing?")
    print("3. Verify your training data has multiple instances per image")
    print("4. Try loading a later checkpoint (e.g., checkpoint-5000 or final_model)")