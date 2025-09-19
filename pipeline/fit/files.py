import pydiffvg
import diffvg
import torch
import cairosvg
import shutil
from subprocess import call, DEVNULL
from pathlib import Path
from datetime import datetime

from .share import sh
from .utils import points_to_svg, points_to_png
import ipdb


def render_fitting_res(shapes, shape_groups, points_n, color_n, save_path):
    shapes[0].points = points_n * sh.w
    shape_groups[0].fill_color = color_n
    # scene_args = pydiffvg.RenderFunction.serialize_scene(\
    #     sh.w, sh.w, shapes, shape_groups, 
    #     filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann, radius = torch.tensor(sh.w/16)))
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        sh.w, sh.w, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(sh.w,   # width
                sh.w,   # height
                2,     # num_samples_x
                2,     # num_samples_y
                102,    # seed
                sh.background, # background_image
                *scene_args)
    # Save the images and differences.
    pydiffvg.imwrite(img.cpu(), save_path / "render.png", gamma=2.2)

    points_to_png(shapes[0].points / sh.w, save_path / "vec.png")

def visualize_video(vis_path, video_path, delete_images):
    # for t in range(sh.epoch):  # adjust number of frames
        # svg_file = vis_path / f"iter_{t:02}.svg"
        # png_file = vis_path / f"vec_{t:02}.png"
        # cairosvg.svg2png(url=str(svg_file), write_to=str(png_file), scale=16)

# Stitch PNGs into video
    call([
        "ffmpeg",
        "-y",
        "-framerate", "20",
        # "-start_number", "",
        "-i", str(vis_path / "iter_%03d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ], stdout=DEVNULL, stderr=DEVNULL)


    if delete_images:
        shutil.rmtree(vis_path)

# create and save exp results

def save_code_snippet(save_path):
    
    package_dir = Path(__file__).parent.parent
    
    save_path = save_path / "@codes"
    save_path.mkdir(exist_ok=True)
    shutil.copytree(package_dir/"fit", save_path/"fit")
    shutil.copytree(package_dir/"prep", save_path/"prep")
    shutil.copytree(package_dir/"visualization", save_path/"visualization")
    shutil.copy(package_dir/"main.py", save_path/"main.py")
    


    # save_path_utils = save_path / "utils"
    # shutil.copytree(package_dir/"utils", save_path_utils)

def create_exp():

    now = datetime.now()
    day_folder = now.strftime("%m-%d")              # e.g., 04-18
    sub_folder = now.strftime("%H-%M-%S")           # e.g., 23-50-01

    full_exp_path = Path(sh.exp_path) / day_folder / sub_folder
    full_exp_path.mkdir(exist_ok=True, parents=True)

    sh.exp_path = full_exp_path  # Update cfg to point to the nested folder

    save_code_snippet(full_exp_path)