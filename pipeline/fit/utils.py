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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import ipdb





def points_to_png(points, png_name,
                  canvas_size=256,
                  stroke="blue", stroke_width=1,
                  point_radius=2, point_color="red",
                  close_path=True):
    """
    Draw [N,2] points directly into a raster PNG.
    Points should be normalized [0,1].
    """

    # Convert torch -> numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {pts.shape}")

    # Scale to pixel coords
    draw_pts = pts * canvas_size
    if close_path:
        draw_pts = np.vstack([draw_pts, draw_pts[0]])


    # Create white background
    img = Image.new("RGB", (canvas_size, canvas_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw polyline
    draw.line([tuple(p) for p in draw_pts], fill=stroke, width=stroke_width)

    # Draw points
    for x, y in draw_pts[:-1] if close_path else draw_pts:
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=point_color
        )

    png_name.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_name)

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

