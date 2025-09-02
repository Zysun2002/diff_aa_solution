import torch
import numpy as np
from .share import sh

def sample_circle(center, radius, n):
    """
    Sample n evenly spaced points along the circumference of a circle.

    Args:
        center (tuple): (cx, cy) of the circle center (constant).
        radius (float): circle radius (constant).
        n (int): number of sample points.

    Returns:
        torch.Tensor: [n, 2] tensor of sampled points (requires_grad=True).
    """
    cx, cy = center

    # Evenly spaced angles (no endpoint=True needed)
    angles = torch.arange(0, n, dtype=torch.float32) * (2 * torch.pi / n)

    # Circle coordinates
    x = cx + radius * torch.cos(angles)
    y = cy + radius * torch.sin(angles)

    points = torch.stack([x, y], dim=1)

    return points.requires_grad_()

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt
import ipdb



def sample_from_boundary(image_path: str) -> torch.nn.Parameter:
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

def points_to_svg(points, svg_name,
                  stroke="blue", stroke_width=1,
                  point_radius=0.6, point_color="red",
                  close_path=True):
    """
    Save [N,2] points as an SVG polyline with optional circles at points.
    Points are written at their given coordinates without normalization.
    """

    from pathlib import Path
    import numpy as np

    try:
        import torch
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
    except ImportError:
        pass

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {pts.shape}")

    draw_pts = pts * sh.w
    if close_path:
        draw_pts = np.vstack([draw_pts, draw_pts[0]])

    # Polyline string
    polyline_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in draw_pts)

    # Build SVG
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sh.w}" height="{sh.w}">']
    svg.append(f'  <polyline points="{polyline_str}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    for x, y in draw_pts[:-1] if close_path else draw_pts:
        svg.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius}" fill="{point_color}"/>')
    svg.append('</svg>')

    svg_str = "\n".join(svg)

    # Ensure parent directory exists
    svg_path = Path(svg_name)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    # Write SVG file
    with svg_path.open("w") as f:
        f.write(svg_str)

