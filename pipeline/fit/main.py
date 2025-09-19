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
import random
import time

from .share import sh
from .utils import points_to_png, points_to_svg
from .files import render_fitting_res, visualize_video, create_exp
from .log import logger, Logger
from .segmentation import sample_from_boundary
from .loss import cal_smooth_loss, cal_curvature_loss, cal_straightness_loss, cal_axis_align_loss, cal_smooth_loss_distance
from .monitor import Monitor

# object-in-subfolder-level

def run(raster_path, exp_path):

    sublogger = Logger()
    # sublogger.create_log(exp_path / "log.txt")

    # prepare
    to_tensor = T.ToTensor()
    raster = to_tensor(Image.open(raster_path).convert("RGBA")).permute(1, 2, 0)
    # ipdb.set_trace()

    pydiffvg.imwrite(raster.cpu(), exp_path / 'target.png', gamma=2.2)

    sh.w = raster.shape[0]
    logger.print(f"current raster: {raster_path}")

    background = torch.zeros((sh.w, sh.w, 4))
    background[..., 3] = 1.0 
    sh.background = background


    # init shape
    ipdb.set_trace()
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    render = pydiffvg.RenderFunction.apply


    # points_n = sample_circle((0.5, 0.5), 0.2, sh.num_samples)

    points_n = sample_from_boundary(exp_path / 'target.png', contour_path=exp_path/'contour.png')
    

    # points_to_svg(points_n, exp_path / "init.svg")

    color_n = torch.tensor(sh.color_guess, requires_grad=True)

    polygon = pydiffvg.Polygon(points = points_n, is_closed = True)
    shapes = [polygon]
    polygon_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = color_n)
    shape_groups = [polygon_group]
    polygon.points = points_n * sh.w
    polygon_group.color = color_n

    optimizer = torch.optim.Adam([points_n], lr=1e-2)
    monitor = Monitor()

    for t in range(sh.epoch):
        
        with monitor.section("forward rendering"):

            optimizer.zero_grad()
            # Forward pass: render the image.
            shapes[0].points = points_n * sh.w
            polygon_group.fill_color = color_n
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                sh.w, sh.w, shapes, shape_groups)
            
            seed = random.randint(0, 2**31 - 1)

            img = render(sh.w,   # width
                        sh.w,   # height
                        2,     # num_samples_x
                        2,     # num_samples_y
                        seed + 117,   # seed
                        sh.background, # background_image
                        *scene_args)
        
        with monitor.section("save_images"):

            # interval = sh.epoch // 10
            interval = 1

            if t >= 0 and t % interval == 0:
            # Save the intermediate render.
                pydiffvg.imwrite(img.cpu(), exp_path / 'vis' / 'render_iter_{:03}.png'.format(t), gamma=2.2)
                points_to_png(points_n, exp_path / 'vis' / 'iter_{:03}.png'.format(t))

            if t == 0:
                pydiffvg.imwrite(img.cpu(), exp_path / 'init_render.png', gamma=2.2)
                points_to_png(points_n, exp_path / "init_vec.png")

            if t == 100:
                pydiffvg.imwrite(img.cpu(), exp_path / 'render_first_stage.png', gamma=2.2)
                points_to_png(points_n, exp_path / "vec_first_stage.png")

        with monitor.section("calculate_loss"):

            img_loss = (img - raster).pow(2).mean() * 10

            points_circle = torch.cat([points_n, points_n[:1]], dim=0)

            smooth_loss =  cal_smooth_loss_distance(points_circle) 

            if t < 100:
                straightness_loss = torch.tensor(0.0)
                axis_align_loss = torch.tensor(0.0)
                curvature_loss = torch.tensor(0.0)
                # straightness_loss = cal_straightness_loss(points_circle) * 0
                # axis_align_loss = cal_axis_align_loss(points_circle) * 0
                # curvature_loss = cal_curvature_loss(points_circle) * 0
            else:
                # straightness_loss = torch.tensor(0.0)
                straightness_loss = torch.tensor(0.0)
                # axis_align_loss = torch.tensor(0.0)
                # curvature_loss = torch.tensor(0.0)
                # curvature_loss = torch.tensor(0.0)
                straightness_loss = cal_straightness_loss(points_circle) * 0
                axis_align_loss = cal_axis_align_loss(points_circle) * 10
                curvature_loss = cal_curvature_loss(points_circle) * 50

            loss = img_loss + smooth_loss + straightness_loss + axis_align_loss + curvature_loss

        with monitor.section("log"):

            if t % 1 == 0:
                logger.print(f'iteration: {t} \n')
                logger.print(f'loss: {loss.item():.6f}, img_loss: {img_loss.item():.6f}, smooth_loss: {smooth_loss.item():.6f} \n')

                sublogger.log_loss(t, img_loss.item(), smooth_loss.item(),\
                                straightness_loss.item(), axis_align_loss.item(),\
                                    curvature_loss.item(), loss.item())

        with monitor.section("backward"):

        # Backpropagate the gradients.
            loss.backward()

            optimizer.step()

    # ipdb.set_trace()
    logger.close()
    sublogger.plot_losses(exp_path / "loss.png")

    monitor.report(exp_path / "time_report.json")
    render_fitting_res(shapes, shape_groups, points_n, color_n, save_path=exp_path)
    visualize_video(exp_path / "vis", exp_path/"vis.mp4", delete_images=False)
    logger.print("-"*40 + "\n\n\n")


def batch(fold, resolution):
    logger.create_log("./log.txt")
    create_exp()

    subfolds = list(fold.glob("*/"))  # materialize generator so tqdm knows length
    for subfold in tqdm(subfolds, desc="curve fitting"):
        
        if not (subfold / "aa_16.png").exists(): continue

        sub_exp_path = sh.exp_path / subfold.name
        sub_exp_path.mkdir(parents=True, exist_ok=True)

        raster_name = f"aa_{resolution}.png"
        shutil.copy(subfold / raster_name, sub_exp_path)

        raster_path = subfold / f"aa_{resolution}.png"

        run(raster_path, sub_exp_path)

if __name__ == "__main__":

    sub_path = Path(r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\09-13\23-19-30\bridge")

    exp_path = sub_path / "temp_axis_align_reg"

    logger.create_log(sub_path / "log.txt")
    
    if exp_path.exists():
        import shutil
        shutil.rmtree(exp_path)
    exp_path.mkdir(parents=True)

    run(raster_path = sub_path / "aa_64.png",\
        exp_path = exp_path)