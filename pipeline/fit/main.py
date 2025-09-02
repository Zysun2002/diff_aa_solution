from pathlib import Path
from PIL import Image
import diffvg
import pydiffvg
import torch
import pydiffvg
import torch
import skimage
from pathlib import Path
import subprocess
import numpy as np
import torchvision.transforms as T
import cairosvg
from svgpathtools import parse_path
from svgpathtools import Path as svgPath
import shutil
import ipdb
from tqdm import tqdm

from .share import sh
from .utils import sample_from_boundary
from .files import render_fitting_res, visualize_video, points_to_svg
from .log import logger


# object-in-subfolder-level

def run(raster_path, exp_path):


    # prepare
    to_tensor = T.ToTensor()
    raster = to_tensor(Image.open(raster_path).convert("RGBA")).permute(1, 2, 0)
    ipdb.set_trace()

    pydiffvg.imwrite(raster.cpu(), exp_path / 'target.png', gamma=2.2)

    sh.w = raster.shape[0]
    logger.print(f"current raster: {raster_path}")


    # init shape
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    render = pydiffvg.RenderFunction.apply


    # points_n = sample_circle((0.5, 0.5), 0.2, sh.num_samples)
    points_n = sample_from_boundary(exp_path / 'target.png')
    


    # ipdb.set_trace()
    points_to_svg(points_n, exp_path / "init.svg")

    color_n = torch.tensor(sh.color_guess, requires_grad=True)

    polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = color_n)
    shape_groups = [polygon_group]
    polygon.points = points_n * sh.w
    polygon_group.color = color_n

    optimizer = torch.optim.Adam([points_n, color_n], lr=1e-2)

    for t in range(sh.epoch):
        optimizer.zero_grad()
        # Forward pass: render the image.
        shapes[0].points = points_n * sh.w
        polygon_group.fill_color = color_n
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            sh.w, sh.w, shapes, shape_groups)
        img = render(sh.w,   # width
                    sh.w,   # height
                    2,     # num_samples_x
                    2,     # num_samples_y
                    t+1,   # seed
                    None, # background_image
                    *scene_args)
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), exp_path / 'vis' / 'iter_{:02}.png'.format(t), gamma=2.2)
        # ipdb.set_trace()
        points_to_svg(points_n, exp_path / 'vis' / 'iter_{:02}.svg'.format(t))

        # Compute the loss function. Here it is L2.
        img_loss = (img - raster).pow(2).sum()

        n = points_n.shape[0]
        idx = torch.arange(n)

        points_prev = points_n[(idx - 1) % n]   # wrap-around for previous
        points_next = points_n[(idx + 1) % n]   # wrap-around for next
        points_curr = points_n

        diff = points_next - 2 * points_curr + points_prev
        smooth_loss = (diff ** 2).sum() * 100

        loss = img_loss + smooth_loss 

        if t % 20 == 0:
            logger.print(f'iteration: {t} \n')
            logger.print(f'loss: {loss.item():.6f}, img_loss: {img_loss.item():.6f}, smooth_loss: {smooth_loss.item():.6f} \n')

        # Backpropagate the gradients.
        loss.backward()
        optimizer.step()

    render_fitting_res(shapes, shape_groups, points_n, color_n, save_path=exp_path/"res")
    visualize_video(exp_path / "vis", exp_path/"vis.mp4", delete_images=True)

    logger.print("-"*40 + "\n\n\n")


def batch(fold, resolution):
    logger.create_log("./log.txt")

    subfolds = list(fold.glob("*/"))  # materialize generator so tqdm knows length
    for subfold in tqdm(subfolds, desc="curve fitting"):
        
        if not (subfold / "aa_16.png").exists(): continue

        exp_path = subfold / str(resolution)
        if exp_path.exists():
            shutil.rmtree(exp_path)
        exp_path.mkdir()

        raster_path = subfold / f"aa_{resolution}.png"

        run(raster_path, exp_path)

if __name__ == "__main__":

    exp_path = Path("/workspace/diffvg/diffAaSolution/data/axe/32")

    logger.create_log("/workspace/diffvg/diffAaSolution/log.txt")
    
    if exp_path.exists():
        import shutil
        shutil.rmtree(exp_path)
    exp_path.mkdir(parents=True)

    run(raster_path = Path("/workspace/diffvg/diffAaSolution/data/axe/aa_32.png"),\
        exp_path = exp_path)