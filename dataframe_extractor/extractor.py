# extractor.py
from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional, Tuple, Union
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from ultralytics import YOLO
from ..efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

from .declare import FrameData
from .helpers import bbox_from_mask, mask_stats

ImageLike = Union[str, np.ndarray, torch.Tensor, Image.Image]


# --------------------------
# I/O helpers
# --------------------------
def to_uint8_rgb(img: ImageLike) -> np.ndarray:
    """Convert path/PIL/numpy/torch to uint8 RGB [H,W,3]."""
    if isinstance(img, str):
        return np.array(Image.open(img).convert("RGB"), dtype=np.uint8)

    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported np shape: {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.5 else arr.clip(0, 255).astype(np.uint8)
        return arr

    if torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # [C,H,W]
            t = t.permute(1, 2, 0)
        if t.ndim != 3:
            raise ValueError(f"Unsupported torch shape: {tuple(t.shape)}")
        t = t.float()
        if t.max().item() <= 1.5:
            t = t * 255.0
        arr = t.clamp(0, 255).byte().numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    raise TypeError(f"Unsupported image type: {type(img)}")


def resize_rgb(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if arr.shape[:2] == (H, W):
        return arr
    return np.array(Image.fromarray(arr).resize((W, H), resample=Image.BILINEAR), dtype=np.uint8)


def ensure_path(img: ImageLike, prefix: str) -> Tuple[str, Optional[str]]:
    """FastSAM is most reliable with file paths."""
    if isinstance(img, str):
        return img, None
    arr = to_uint8_rgb(img)
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    Image.fromarray(arr).save(tmp_path)
    return tmp_path, tmp_path


def gray_tensor(img: ImageLike, hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Return grayscale float tensor [H,W] in [0,1]."""
    arr = resize_rgb(to_uint8_rgb(img), hw)
    g = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32) / 255.0
    return torch.from_numpy(g).to(device)


# --------------------------
# Frame extractor
# --------------------------
class FrameExtractor:
    """
    Owns the YOLO + EfficientSAM models and turns an image into FrameData.
    """

    def __init__(
        self,
        *,
        yolo_weights: str,
        sam_weights: str,
        sam_type: str = "vitt",
        device: Optional[str] = None,
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.8,
        retina_masks: bool = True,  # kept for compatibility, not used
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.yolo_model = YOLO(yolo_weights)
        self.sam_model = self._load_efficient_sam(sam_type, sam_weights)
        
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.sam_type = sam_type
    
    def _load_efficient_sam(self, model_type: str, model_path: str):
        """Load EfficientSAM model."""
        model_path = Path(model_path)
        
        if model_path.suffix == '.zip' or str(model_path).endswith('.pt.zip'):
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(model_path.parent)
            model_path = model_path.parent / model_path.stem
            if not model_path.suffix == '.pt':
                model_path = Path(str(model_path) + '.pt')
        
        if not model_path.exists():
            raise FileNotFoundError(f"EfficientSAM model not found: {model_path}")
        
        if model_type == 'vitt':
            model = build_efficient_sam_vitt(checkpoint=str(model_path))
        elif model_type == 'vits':
            model = build_efficient_sam_vits(checkpoint=str(model_path))
        else:
            raise ValueError(f"Unknown SAM type: {model_type}")
        
        model = model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def segment_to_framedata(
        self,
        name: str,
        img: ImageLike,
        hw_ref: Tuple[int, int],
        *,
        min_area_keep: int,
    ) -> FrameData:
        original_rgb = to_uint8_rgb(img)
        orig_h, orig_w = original_rgb.shape[:2]
        
        rgb = resize_rgb(original_rgb, hw_ref)
        H, W = rgb.shape[:2]
        gray = gray_tensor(rgb, (H, W), self.device)
        
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        
        # Step 1: YOLO Detection on original image
        yolo_results = self.yolo_model.predict(
            original_bgr,
            conf=self.conf,
            device=str(self.device),
            verbose=False
        )
        
        # Extract bounding boxes
        if len(yolo_results) == 0 or yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
            # No detections - return empty FrameData
            masks = torch.empty((0, H, W), dtype=torch.bool, device=self.device)
            area = torch.empty((0,), device=self.device)
            cent = torch.empty((0, 2), device=self.device)
            bbox: List[Optional[Tuple[int, int, int, int]]] = []
            return FrameData(name=name, gray=gray, masks=masks, area=area, centroid=cent, bbox=bbox)
        
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # xyxy format in original coordinates
        
        # Scale boxes to resized dimensions if needed
        if (orig_h, orig_w) != (H, W):
            scale_x = W / orig_w
            scale_y = H / orig_h
            boxes[:, [0, 2]] *= scale_x  # x1, x2
            boxes[:, [1, 3]] *= scale_y  # y1, y2
        
        # Convert resized RGB to BGR for EfficientSAM
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Step 2: EfficientSAM with scaled box prompts (batched)
        mask_list = self._run_efficient_sam_batched(bgr, boxes)
        
        # Convert to torch tensor [N, H, W]
        if len(mask_list) > 0:
            masks = torch.stack([torch.from_numpy(m).to(self.device) for m in mask_list])
        else:
            masks = torch.empty((0, H, W), dtype=torch.bool, device=self.device)
        
        # Sort by area desc, drop tiny masks
        if masks.numel() > 0:
            area, cent = mask_stats(masks)
            keep = area >= float(min_area_keep)
            masks = masks[keep]
            area = area[keep]
            cent = cent[keep]
            if masks.numel() > 0:
                order = torch.argsort(area, descending=True)
                masks = masks[order]
                area = area[order]
                cent = cent[order]
        else:
            area = torch.empty((0,), device=self.device)
            cent = torch.empty((0, 2), device=self.device)
        
        bbox_list: List[Optional[Tuple[int, int, int, int]]] = [bbox_from_mask(m) for m in masks]
        
        return FrameData(name=name, gray=gray, masks=masks, area=area, centroid=cent, bbox=bbox_list)
    
    def _run_efficient_sam_batched(self, image_bgr: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """Run EfficientSAM with batched box prompts."""
        orig_h, orig_w = image_bgr.shape[:2]
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image_rgb).to(self.device)
        
        num_boxes = len(boxes)
        all_points = []
        valid_indices = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(float)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, orig_w - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            all_points.append([[x1, y1], [x2, y2]])
            valid_indices.append(i)
        
        # No valid boxes
        if len(all_points) == 0:
            return []
        
        # Batched inference
        input_points = torch.tensor([all_points], dtype=torch.float32).to(self.device)
        input_labels = torch.tensor([[[2, 3]] * len(all_points)]).to(self.device)
        
        try:
            with torch.no_grad():
                predicted_logits, predicted_iou = self.sam_model(
                    image_tensor[None, ...],
                    input_points,
                    input_labels,
                )
            
            # Extract masks
            masks = []
            for i in range(len(valid_indices)):
                # Get best mask by IoU
                sorted_ids = torch.argsort(predicted_iou[0, i], dim=-1, descending=True)
                best_mask_idx = sorted_ids[0]
                
                mask = torch.ge(predicted_logits[0, i, best_mask_idx, :, :], 0).cpu().numpy()
                masks.append(mask.astype(bool))
            
            return masks
            
        except Exception as e:
            print(f"Warning: EfficientSAM batched inference failed: {e}")
            # Fallback: simple box masks
            masks = []
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(x1, orig_w - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                mask = np.zeros((orig_h, orig_w), dtype=bool)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
                masks.append(mask)
            
            return masks