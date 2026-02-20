# model/helpers.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import math

from .declare import BBox


# --------------------------
# Geometry + bbox helpers
# --------------------------

def mask_stats(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    masks: [N,H,W] bool on device
    returns:
      area: [N] float
      centroid: [N,2] float (cx,cy)
    """
    if masks.numel() == 0:
        return torch.empty((0,), device=masks.device), torch.empty((0, 2), device=masks.device)

    m = masks.float()
    area = m.sum(dim=(1, 2)).clamp_min(1.0)

    N, H, W = masks.shape
    xs = torch.arange(W, device=masks.device, dtype=torch.float32)
    ys = torch.arange(H, device=masks.device, dtype=torch.float32)

    mx = m.sum(dim=1)  # [N,W] sum over y
    my = m.sum(dim=2)  # [N,H] sum over x

    cx = (mx * xs[None, :]).sum(dim=1) / area
    cy = (my * ys[None, :]).sum(dim=1) / area
    cent = torch.stack([cx, cy], dim=1)
    return area, cent


def bbox_from_mask(mask: torch.Tensor) -> Optional[BBox]:
    """mask: [H,W] bool -> (y0,y1,x0,x1) exclusive."""
    if mask.numel() == 0 or not mask.any():
        return None
    ys = torch.where(mask.any(dim=1))[0]
    xs = torch.where(mask.any(dim=0))[0]
    y0 = int(ys[0].item()); y1 = int(ys[-1].item()) + 1
    x0 = int(xs[0].item()); x1 = int(xs[-1].item()) + 1
    return y0, y1, x0, x1


def crop2d_zeropad(img: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    """img: [H,W] -> [y1-y0, x1-x0], zeros outside bounds."""
    H, W = img.shape
    out = torch.zeros((y1 - y0, x1 - x0), device=img.device, dtype=img.dtype)

    sy0, sy1 = max(0, y0), min(H, y1)
    sx0, sx1 = max(0, x0), min(W, x1)

    dy0, dx0 = sy0 - y0, sx0 - x0
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# --------------------------
# Morphology
# --------------------------
def smooth_shrink_mask(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    dilate_px = erode_px
    erode_px = 2 * erode_px
    m = mask.float()[None, None]

    # erosion
    if erode_px > 0:
        k = 2 * erode_px + 1
        m = F.pad(m, (erode_px, erode_px, erode_px, erode_px), value=0.0)
        m = -F.max_pool2d(-m, kernel_size=k, stride=1)

    # dilation (weaker)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = F.pad(m, (dilate_px, dilate_px, dilate_px, dilate_px), value=0.0)
        m = F.max_pool2d(m, kernel_size=k, stride=1)

    out = (m[0,0] > 0.5)
    return out

# --------------------------
# IoU + overlap coefficient (separate)
# --------------------------

def cross_normalize_and_contrast(
    x1: torch.Tensor,
    x2: torch.Tensor,
    intensity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Jointly normalize and apply contrast to two tensors.

    x1, x2: tensors with same shape (e.g. [H,W] or [1,1,H,W])
    intensity: contrast gain (1.0 = no change)

    Returns:
        x1_out, x2_out
    """
    # joint min / max
    joint_min = torch.min(x1.min(), x2.min())
    joint_max = torch.max(x1.max(), x2.max())
    rng_raw = joint_max - joint_min
    if float(rng_raw.item()) <= 0.25:
        return x1, x2
    rng = rng_raw.clamp_min(1e-6)

    # normalize to [0,1] using shared range
    x1n = (x1 - joint_min) / rng
    x2n = (x2 - joint_min) / rng

    if intensity != 1.0:
        x1n = (x1n - 0.5) * intensity + 0.5
        x2n = (x2n - 0.5) * intensity + 0.5
        x1n = x1n.clamp(0.0, 1.0)
        x2n = x2n.clamp(0.0, 1.0)

    return x1n, x2n
def iou_unaligned(A: torch.Tensor, B: torch.Tensor) -> float:
    """IoU with NO alignment. A,B: [H,W] bool."""
    inter = int((A & B).sum().item())
    a = int(A.sum().item())
    b = int(B.sum().item())
    union = a + b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def iou_thr_from_area(
    area: int,
    *,
    base: float,
    relax_max: float,
    area_ref: int,
) -> float:
    """
    For large masks (>= area_ref), threshold ~ base.
    For tiny masks, threshold relaxes down to base - relax_max.
    """
    s = min(1.0, (area / area_ref) ** 0.5)
    thr = base - relax_max * (1.0 - s)
    return max(0.0, min(base, thr))


def overlap_coef_aligned(A: torch.Tensor, B: torch.Tensor, dy: int, dx: int) -> float:
    """
    Overlap coefficient after aligning B into A coords by (dy,dx):
      |A ∩ shift(B)| / min(|A|, |shift(B)|_inbounds)

    Uses in-bounds area of shifted B (no explicit shifting tensor).
    """
    H, W = A.shape

    # slices for overlap without shifting
    if dy >= 0:
        yA0, yA1 = dy, H
        yB0, yB1 = 0, H - dy
    else:
        yA0, yA1 = 0, H + dy
        yB0, yB1 = -dy, H

    if dx >= 0:
        xA0, xA1 = dx, W
        xB0, xB1 = 0, W - dx
    else:
        xA0, xA1 = 0, W + dx
        xB0, xB1 = -dx, W

    if (yA1 <= yA0) or (xA1 <= xA0) or (yB1 <= yB0) or (xB1 <= xB0):
        return 0.0

    Aroi = A[yA0:yA1, xA0:xA1]
    Broi = B[yB0:yB1, xB0:xB1]

    inter = int((Aroi & Broi).sum().item())
    a = int(A.sum().item())
    b_in = int(Broi.sum().item())
    denom = min(a, b_in)
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


# --------------------------
# Pixel-wise maps + masked stats
# --------------------------
def ncc_map(img1: torch.Tensor, img2: torch.Tensor, patch: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    img1,img2: [1,1,H,W] in [0,1]
    returns (1-ncc)/2 in [0,1], shape [1,1,H,W]
    """
    if patch % 2 == 0:
        raise ValueError("patch must be odd")
    r = patch // 2
    i1 = F.pad(img1, (r, r, r, r), mode="reflect")
    i2 = F.pad(img2, (r, r, r, r), mode="reflect")

    mu1 = F.avg_pool2d(i1, kernel_size=patch, stride=1)
    mu2 = F.avg_pool2d(i2, kernel_size=patch, stride=1)

    mu11 = F.avg_pool2d(i1 * i1, kernel_size=patch, stride=1)
    mu22 = F.avg_pool2d(i2 * i2, kernel_size=patch, stride=1)
    mu12 = F.avg_pool2d(i1 * i2, kernel_size=patch, stride=1)

    var1 = (mu11 - mu1 * mu1).clamp_min(0.0)
    var2 = (mu22 - mu2 * mu2).clamp_min(0.0)
    cov12 = mu12 - mu1 * mu2

    ncc = cov12 / (torch.sqrt(var1 * var2) + eps)
    ncc = ncc.clamp(-1.0, 1.0)
    return (1.0 - ncc) * 0.5


def census_diff_map_ternary(
    img1: torch.Tensor,
    img2: torch.Tensor,
    patch: int = 3,
    eps: float = 0.01,
) -> torch.Tensor:
    """
    Ternary census diff with dead-zone epsilon.

    img1,img2: [1,1,H,W] in [0,1]
    patch: odd >= 3
    eps: dead-zone threshold in [0,1]

    For each neighbor value p and center c, encode ternary state:
        -1 if p > c + eps
         0 if |p - c| <= eps
        +1 if p < c - eps
    (equivalently, based on d = c - p)

    Diff metric: normalized mismatch rate across neighbors (excluding center):
        diff = mean( state1 != state2 )  in [0,1]

    Returns: [1,1,H,W] float in [0,1]
    """
    if patch % 2 == 0 or patch < 3:
        raise ValueError("patch must be odd and >=3")
    if eps < 0:
        raise ValueError("eps must be >= 0")

    r = patch // 2
    P = patch * patch
    center = P // 2

    # [1, P, H*W]
    p1 = F.unfold(img1, kernel_size=patch, padding=r)
    p2 = F.unfold(img2, kernel_size=patch, padding=r)

    # [1, 1, H*W]
    c1 = p1[:, center:center + 1, :]
    c2 = p2[:, center:center + 1, :]

    # differences to center: d = c - p  (positive => neighbor darker than center)
    d1 = c1 - p1  # [1,P,H*W]
    d2 = c2 - p2

    # ternary encode: +1 if d > eps, 0 if |d|<=eps, -1 if d < -eps
    # (we use int8 for compactness)
    t1 = torch.zeros_like(d1, dtype=torch.int8)
    t2 = torch.zeros_like(d2, dtype=torch.int8)

    t1[d1 >  eps] =  1
    t1[d1 < -eps] = -1
    t2[d2 >  eps] =  1
    t2[d2 < -eps] = -1

    # drop center element
    keep = torch.ones((P,), device=img1.device, dtype=torch.bool)
    keep[center] = False

    # mismatch rate across neighbors
    diff = (t1[:, keep, :] != t2[:, keep, :]).float().mean(dim=1)  # [1, H*W]

    H, W = img1.shape[-2:]
    return diff.view(1, 1, H, W)


def masked_mean_var(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> Tuple[float, float]:
    """
    x: [1,1,H,W] float
    m: [1,1,H,W] float mask in {0,1}
    returns: (mean, var) over masked pixels; if empty -> (+inf, +inf)
    """
    denom = m.sum().clamp_min(0.0)
    if float(denom.item()) < 0.5:
        return float("inf"), float("inf")

    mean = (x * m).sum() / (denom + eps)
    var = ((x - mean) ** 2 * m).sum() / (denom + eps)
    return float(mean.item()), float(var.item())


@torch.no_grad()
def masked_mean_var_ignore_one_local_peak(
    x: torch.Tensor,          # [1,1,H,W] float
    m: torch.Tensor,          # [1,1,H,W] float mask in {0,1}
    eps: float = 1e-6,
    area_frac: float = 0.1,
) -> Tuple[float, float]:
    """
    Same API as masked_mean_var, but ignores ONE localized peak region using
    fast box-sums via integral images (summed-area tables).

    Strategy:
      1) Compute masked mean
      2) Highlight pixels > (2/3)*mean within mask
      3) For 3 shapes (vertical rect, square, horizontal rect) with area ~= area_frac * mask_area:
         - find window offset maximizing highlighted count
         - only consider windows that are mostly inside the mask
      4) Remove ONLY highlighted pixels inside the chosen window
      5) Return mean/var over remaining masked pixels
    """
    assert x.ndim == 4 and m.ndim == 4 and x.shape == m.shape
    H, W = x.shape[-2:]

    mask = (m > 0.5)
    denom = mask.sum()
    if float(denom.item()) < 0.5:
        return float("inf"), float("inf")
    mask_f = mask.float()

    # --- 1) masked mean
    mean = (x * mask_f).sum() / (denom + eps)

    # --- 2) highlight map
    thr = (2.0 / 3.0) * mean
    highlight = ((x > thr) & mask).float()

    if float(highlight.sum().item()) < 0.5:
        var = ((x - mean) ** 2 * mask_f).sum() / (denom + eps)
        return float(mean.item()), float(var.item())

    # --- 3) determine window area K
    total_area = int(denom.item())
    K = max(1, int(area_frac * total_area))

    # Build 5 shapes with area ~ K
    s = max(1, int(round(math.sqrt(K))))
    square = (s, max(1, int(math.ceil(K / s))))

    h_small = max(1, s // 2)
    w_big = max(1, int(math.ceil(K / h_small)))
    horizontal = (h_small, w_big)
    vertical = (w_big, h_small)

    h_mid = max(1, int(round(0.75 * s)))
    w_mid = max(1, int(math.ceil(K / h_mid)))
    horizontal_mid = (h_mid, w_mid)
    vertical_mid = (w_mid, h_mid)

    shapes = []
    for kh, kw in [vertical, vertical_mid, square, horizontal_mid, horizontal]:
        if kh <= H and kw <= W and (kh, kw) not in shapes:
            shapes.append((kh, kw))

    # --- integral image helper for all sliding window sums
    def box_sum_all(img: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
        """
        img: [1,1,H,W]
        returns: [1,1,H-kh+1,W-kw+1] where each entry is sum over khxkw window
        """
        # integral image with zero padding at top/left
        ii = img.cumsum(dim=-2).cumsum(dim=-1)  # [1,1,H,W]
        ii = torch.nn.functional.pad(ii, (1, 0, 1, 0), mode="constant", value=0.0)  # [1,1,H+1,W+1]

        # rectangle sum using 4 corners
        # S(y,x) = ii[y+kh, x+kw] - ii[y, x+kw] - ii[y+kh, x] + ii[y, x]
        return (
            ii[..., kh:, kw:]
            - ii[..., :-kh, kw:]
            - ii[..., kh:, :-kw]
            + ii[..., :-kh, :-kw]
        )

    best_score = None
    best = None

    # Require window to lie mostly inside mask (same behavior as before)
    valid_frac = 0.95

    for kh, kw in shapes:
        score = box_sum_all(highlight, kh, kw)      # count of highlighted pixels in window
        mask_cnt = box_sum_all(mask_f, kh, kw)      # count of masked pixels in window

        valid = mask_cnt >= (valid_frac * (kh * kw))
        score = torch.where(valid, score, torch.full_like(score, -1e9))

        vmax = score.max()
        if best_score is None or float(vmax.item()) > float(best_score.item()):
            idx = int(score.view(-1).argmax().item())
            outW = score.shape[-1]
            top = idx // outW
            left = idx % outW
            best_score = vmax
            best = (kh, kw, top, left)

    if best is None or float(best_score.item()) < -1e8:
        var = ((x - mean) ** 2 * mask_f).sum() / (denom + eps)
        return float(mean.item()), float(var.item())

    kh, kw, top, left = best

    # --- 4) build window mask and remove ONLY highlighted pixels within it
    win = torch.zeros_like(mask)
    win[..., top:top + kh, left:left + kw] = True

    new_mask = mask & ~(win & (highlight > 0.5))
    new_mask_f = new_mask.float()

    denom2 = new_mask_f.sum()
    if float(denom2.item()) < 0.5:
        return float("inf"), float("inf")

    mean2 = (x * new_mask_f).sum() / (denom2 + eps)
    var2 = ((x - mean2) ** 2 * new_mask_f).sum() / (denom2 + eps)
    return float(mean2.item()), float(var2.item())