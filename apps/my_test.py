import pydiffvg
import diffvg
import ipdb
import torch
from pathlib import Path
import subprocess

# Use GPU if available

output_path = Path("results/low_res_two_overlapped_circles")

pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_size = 32
circle1 = pydiffvg.Circle(radius = torch.tensor(4.0),
                         center = torch.tensor([22.0, 22.0]))
circle2 = pydiffvg.Circle(radius = torch.tensor(3.0),
                         center = torch.tensor([24.0, 24.0]))
shapes = [circle1, circle2]
circle_group1 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([1, 0., 0., 1.0]))
circle_group2 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([1]),
    fill_color = torch.tensor([0, 1., 0., 1.0]))
shape_groups = [circle_group1, circle_group2]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_size,
    canvas_height=canvas_size,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(5.0)))

render = pydiffvg.RenderFunction.apply
img = render(canvas_size, # width
             canvas_size, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), output_path / 'target.png', gamma=2.2)
target = img.clone()


radius_pred_1 = torch.tensor(3.0 / canvas_size, requires_grad=True)
center_pred_1 = torch.tensor([20.0 / canvas_size , 20.0 / canvas_size], requires_grad=True)
color_pred_1 = torch.tensor([0.5, 0.1, 0.1, 1.0], requires_grad=True)
shape_1 = pydiffvg.Circle(radius = radius_pred_1 * canvas_size, center = center_pred_1 * canvas_size)
circle_pred_1 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]), fill_color = color_pred_1)


radius_pred_2 = torch.tensor(5.0 / canvas_size, requires_grad=True)
center_pred_2 = torch.tensor([26.0 / canvas_size , 26.0 / canvas_size], requires_grad=True)
color_pred_2 = torch.tensor([0.1, 0.5, 0.1, 1.0], requires_grad=True)
shape_2 = pydiffvg.Circle(radius = radius_pred_2 * canvas_size, center = center_pred_2 * canvas_size)
circle_pred_2 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([1]), fill_color = color_pred_2)

shapes_pred = [shape_1, shape_2]
shape_groups_pred = [circle_pred_1, circle_pred_2]

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_size, canvas_size, shapes_pred, shape_groups_pred, \
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(5.0)))
img = render(canvas_size, # width
             canvas_size, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), output_path / 'init.png', gamma=2.2)


# Run 100 Adam iterations.
optimizer = torch.optim.Adam([radius_pred_1, center_pred_1, color_pred_1, radius_pred_2, center_pred_2, color_pred_2], lr=1e-2)

for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    shape_1.radius = radius_pred_1 * canvas_size
    shape_1.center = center_pred_1 * canvas_size
    circle_pred_1.fill_color = color_pred_1

    shape_2.radius = radius_pred_2 * canvas_size
    shape_2.center = center_pred_2 * canvas_size
    circle_pred_2.fill_color = color_pred_2

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_size, canvas_size, shapes_pred, shape_groups_pred, \
            filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(5.0)))
    
    # ipdb.set_trace()
    img = render(canvas_size,   # width
                 canvas_size,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), output_path / 'iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('radius.grad:', radius_pred_1.grad, radius_pred_2.grad)
    print('center.grad:', center_pred_1.grad, center_pred_2.grad)
    print('color.grad:', color_pred_1.grad, color_pred_2.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    # print('radius:', circle.radius)
    # print('center:', circle.center)
    # print('color:', circle_group.fill_color)


scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_size, canvas_size, shapes, shape_groups, \
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(5.0)))
img = render(canvas_size,   # width
             canvas_size,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None,
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), output_path / 'final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    output_path / "iter_%d.png", "-vb", "20M",
    output_path / "out.mp4"])

command = [
    "ffmpeg", "-y", "-i", output_path / "out.mp4",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-strict", "experimental",
    "-tune", "fastdecode",
    "-pix_fmt", "yuv420p",
    "-b:a", "192k",
    "-ar", "48000",
    output_path / "output.mp4"
]

subprocess.run(command, check=True)