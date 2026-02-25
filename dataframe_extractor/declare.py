# model/declare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

BBox = Tuple[int, int, int, int]  # (y0, y1, x0, x1) exclusive


@dataclass
class FrameData:
    """
    A segmented frame packaged for matching.
    """
    name: str
    gray: torch.Tensor              # [H,W] float32 in [0,1]
    masks: torch.Tensor             # [N,H,W] bool
    area: torch.Tensor              # [N] float32
    centroid: torch.Tensor          # [N,2] float32 (cx,cy)
    bbox: List[Optional[BBox]]      # length N


@dataclass
class Match:
    """
    Pairwise matchability record between:
      - GT mask (mask1) at gt_idx
      - Other-frame mask (mask2) at idx

    level:
      0 = no match (usually represented as None)
      1 = correspondence found (augment not proven)
      2 = structural consistency proved
      3 = appearance entire change proved
    """
    frame: str
    idx: int
    dy: int
    dx: int

    area1: float
    area2: float

    centroid1: Tuple[float, float]
    centroid2: Tuple[float, float]

    overlap_coef: float            # aligned overlap coefficient (after shifting by dy/dx)
    iou: float                     # unaligned IoU (dy=dx=0)

    ncc_mean: float
    ncc_var: float
    census_mean: float
    census_var: float
    appearance_diff: float

    level: int                     # 1, 2, or 3