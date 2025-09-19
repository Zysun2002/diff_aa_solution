import numpy as np
import cv2
from PIL import Image
import torch
import ipdb
from .share import sh

def sample_from_boundary(image_path, contour_path=None):
    """
    Extract evenly spaced boundary points from an image containing
    a single closed object, normalized to [0,1].

    Args:
        image_path (str): path to the raster image
        num_points (int): number of points to sample along the boundary

    Returns:
        torch.nn.Parameter: (num_points, 2) normalized (x,y) points
                            with requires_grad=True
    """
    # === Step 1: Load and grayscale ===
    img = np.array(Image.open(image_path).convert("L"))
    h, w = img.shape  # height, width
    
    # === Step 2: Binarize ===
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # === Step 3: Find contour ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in the image")
    contour = max(contours, key=cv2.contourArea).squeeze()  # (N, 2)

    if contour_path: 
        # create a blank black image
        contour_img = np.zeros((h, w, 3), dtype=np.uint8)
        # draw contour in white (thickness = 2 px)
        cv2.drawContours(contour_img, [contour.reshape(-1, 1, 2)], -1, (255, 255, 255), 1)
        # save as PNG
        cv2.imwrite(contour_path, contour_img)


    # === Step 4: Compute cumulative arc-length ===
    diffs = np.diff(contour, axis=0, append=contour[:1])
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumlen = np.cumsum(segment_lengths)
    cumlen = np.insert(cumlen, 0, 0)
    total_len = cumlen[-1]

    # === Step 5: Sample evenly spaced points ===
    target_lens = np.linspace(0, total_len, sh.num_samples, endpoint=False)
    sampled_points = []
    for t in target_lens:
        idx = np.searchsorted(cumlen, t) - 1
        idx = np.clip(idx, 0, len(contour) - 1)

        seg_start, seg_end = contour[idx], contour[(idx + 1) % len(contour)]
        seg_len = segment_lengths[idx]

        if seg_len == 0:
            sampled_points.append(seg_start)
        else:
            alpha = (t - cumlen[idx]) / seg_len
            pt = (1 - alpha) * seg_start + alpha * seg_end
            sampled_points.append(pt)

    sampled_points = np.array(sampled_points, dtype=np.float32)  # (num_points, 2)

    # === Step 6: Normalize to [0,1] ===
    sampled_points[:, 0] /= w   # normalize x by width
    sampled_points[:, 1] /= h   # normalize y by height

    # === Step 7: Return as nn.Parameter ===
    return torch.nn.Parameter(torch.tensor(sampled_points, dtype=torch.float32))
