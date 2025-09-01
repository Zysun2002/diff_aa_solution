import pydiffvg
import torch
import skimage
from pathlib import Path
import subprocess
import ipdb
import numpy as np
import cairosvg
from svgpathtools import parse_path
from svgpathtools import Path as svgPath


import numpy as np


import torch


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


def points_to_svg(points, svg_name,
                  width=64, height=64, 
                  stroke="blue", stroke_width=1, 
                  point_radius=0.6, point_color="red", 
                  normalize=True, padding=5,
                  close_path=True):
    """
    Convert [N,2] points into an SVG polyline with highlighted points.
    If close_path=True, connects the last point back to the first.
    """
    try:
        import torch
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
    except ImportError:
        pass

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {pts.shape}")

    if normalize:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        dx = dx if dx > 0 else 1
        dy = dy if dy > 0 else 1
        sx = (width - 2*padding) / dx
        sy = (height - 2*padding) / dy
        s = min(sx, sy)
        norm = np.zeros_like(pts)
        norm[:, 0] = (pts[:, 0] - xmin) * s + padding
        norm[:, 1] = (pts[:, 1] - ymin) * s + padding
        draw_pts = norm
    else:
        draw_pts = pts

    if close_path:
        # Append first point again at the end
        draw_pts = np.vstack([draw_pts, draw_pts[0]])

    # Polyline string
    polyline_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in draw_pts)

    # Build SVG
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.append(f'  <polyline points="{polyline_str}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    for x, y in draw_pts[:-1] if close_path else draw_pts:
        svg.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius}" fill="{point_color}"/>')
    svg.append('</svg>')

    svg_str = "\n".join(svg)
    with open(svg_name, "w") as f:
        f.write(svg_str)


# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
exp_path = Path("results/usePolylinesToFitPathWith32Vertex")

path_str = 'M305.3,295.9c-7.1,0-13.7-3-18.8-8.1c-8.6-8.7-22.9-7.9-30.6,1.6c-7.7-9.5-21.9-10.2-30.6-1.6\
		c-5,5-11.7,8.1-18.8,8.1c-3.2,0-5.6,2.9-4.9,6.1c5.1,25.5,37.8,33.7,54.3,13.3c16.5,20.4,49.2,12.2,54.3-13.3\
		C310.9,298.8,308.5,295.9,305.3,295.9z'
path = parse_path(path_str)

# Get bounding box (xmin, xmax, ymin, ymax)
xmin, xmax, ymin, ymax = path.bbox()
# Desired new top-left corner
target_x, target_y = 12, 32

# Compute translation
dx = target_x - xmin
dy = target_y - ymin

# Apply translation
moved_path = path.translated(complex(dx, dy))

scale_factor = 0.5
path = moved_path.scaled(scale_factor)
new_path_str = path.d()
# print(new_path_str)


canvas_size = 64
# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path(new_path_str)
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_size, canvas_size, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(canvas_size, # width
             canvas_size, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), exp_path / 'target.png', gamma=2.2)
target = img.clone()
ipdb.set_trace()


# init shapes for fitting
points_n = sample_circle((0.5, 0.5), 0.5, 32)
color_n = torch.tensor([0.6, 0.8, 0.3, 1.0], requires_grad=True)

polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
shapes = [polygon]
polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                    fill_color = color_n)
shape_groups = [polygon_group]


polygon.points = points_n * canvas_size
polygon_group.color = color_n
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_size, canvas_size, shapes, shape_groups)
img = render(canvas_size, # width
             canvas_size, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), exp_path / 'init.png', gamma=2.2)
points_to_svg(shapes[0].points, exp_path / "init.svg")


# Optimize
optimizer = torch.optim.Adam([points_n, color_n], lr=1e-2)
# Run 100 Adam iterations.
for t in range(300):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    shapes[0].points = points_n * canvas_size
    polygon_group.fill_color = color_n
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_size, canvas_size, shapes, shape_groups)
    img = render(canvas_size,   # width
                 canvas_size,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), exp_path / 'iter_{:02}.png'.format(t), gamma=2.2)
    points_to_svg(shapes[0].points, exp_path / 'iter_{:02}.svg'.format(t))


    # Compute the loss function. Here it is L2.
    img_loss = (img - target).pow(2).sum()

    n = points_n.shape[0]
    idx = torch.arange(n)

    points_prev = points_n[(idx - 1) % n]   # wrap-around for previous
    points_next = points_n[(idx + 1) % n]   # wrap-around for next
    points_curr = points_n

    diff = points_next - 2 * points_curr + points_prev
    smooth_loss = (diff ** 2).sum() * 500

    loss = img_loss + smooth_loss 

    print(f'loss: {loss.item():.6f}, img_loss: {img_loss.item():.6f}, smooth_loss: {smooth_loss.item():.6f}')

    # Backpropagate the gradients.
    loss.backward()
    optimizer.step()


# Render the final result.
shapes[0].points = points_n * canvas_size
polygon_group.fill_color = color_n
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_size, canvas_size, shapes, shape_groups)
img = render(canvas_size,   # width
             canvas_size,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), exp_path / 'final.png')
points_to_svg(shapes[0].points, "final.svg")

# Convert the intermediate renderings to a video.
from subprocess import call
# call(["ffmpeg", "-framerate", "20", "-i",
#     exp_path / "iter_%02d.svg", "-vb", "20M",
#     exp_path / "out_svg.mp4"])

# command = [
#     "ffmpeg", "-y", "-i", exp_path / "out_svg.mp4",
#     "-c:v", "libx264",
#     "-c:a", "aac",
#     "-strict", "experimental",
#     "-tune", "fastdecode",
#     "-pix_fmt", "yuv420p",
#     "-b:a", "192k",
#     "-ar", "48000",
#     exp_path / "output.mp4"
# ]
for t in range(300):  # adjust number of frames
    svg_file = exp_path / f"iter_{t:02}.svg"
    png_file = exp_path / f"iter_{t:02}_from_svg.png"
    cairosvg.svg2png(url=str(svg_file), write_to=str(png_file), scale=16)
    # print(f"Converted {svg_file} → {png_file}")

# Stitch PNGs into video
call([
    "ffmpeg",
    "-y",
    "-framerate", "20",
    "-i", str(exp_path / "iter_%02d_from_svg.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(exp_path / "output.mp4")
])

# subprocess.run(command, check=True)
