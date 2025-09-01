import pydiffvg
import torch
import skimage

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 510, 510
# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path('M305.3,295.9c-7.1,0-13.7-3-18.8-8.1c-8.6-8.7-22.9-7.9-30.6,1.6c-7.7-9.5-21.9-10.2-30.6-1.6\
		c-5,5-11.7,8.1-18.8,8.1c-3.2,0-5.6,2.9-4.9,6.1c5.1,25.5,37.8,33.7,54.3,13.3c16.5,20.4,49.2,12.2,54.3-13.3\
		C310.9,298.8,308.5,295.9,305.3,295.9z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(510, # width
             510, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_path/target.png', gamma=2.2)
target = img.clone()

# Move the path to produce initial guess
# normalize points for easier learning rate
noise = torch.FloatTensor(shapes[0].points.shape).uniform_(0.0, 1.0)
points_n = (shapes[0].points.clone() + (noise * 60 - 30)) / 510.0
points_n.requires_grad = True
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
shapes[0].points = points_n * 510
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(510, # width
             510, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_path/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    shapes[0].points = points_n * 510
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(510,   # width
                 510,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_path/iter_{:02}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', shapes[0].points)
    print('color:', path_group.fill_color)

# Render the final result.
shapes[0].points = points_n * 510
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(510,   # width
             510,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_path/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "20", "-i",
    "results/single_path/iter_%02d.png", "-vb", "20M",
    "results/single_path/out.mp4"])
