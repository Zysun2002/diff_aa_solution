import pydiffvg
import diffvg
import torch
import cairosvg
import shutil
from subprocess import call, DEVNULL


from .share import sh
from .utils import points_to_svg
import ipdb


def render_fitting_res(shapes, shape_groups, points_n, color_n, save_path):
    shapes[0].points = points_n * sh.w
    shape_groups[0].fill_color = color_n
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        sh.w, sh.w, shapes, shape_groups, 
        filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann, radius = torch.tensor(sh.w/16)))
    

    background = torch.zeros((sh.w, sh.w, 4))
    background[..., 3] = 1.0 
    render = pydiffvg.RenderFunction.apply
    img = render(sh.w,   # width
                sh.w,   # height
                2,     # num_samples_x
                2,     # num_samples_y
                102,    # seed
                background, # background_image
                *scene_args)
    # Save the images and differences.
    pydiffvg.imwrite(img.cpu(), save_path.with_suffix(".png"))

    points_to_svg(shapes[0].points / sh.w, save_path.with_suffix(".svg"))

def visualize_video(vis_path, video_path, delete_images):
    for t in range(sh.epoch):  # adjust number of frames
        svg_file = vis_path / f"iter_{t:02}.svg"
        png_file = vis_path / f"iter_{t:02}_from_svg.png"
        cairosvg.svg2png(url=str(svg_file), write_to=str(png_file), scale=16)

# Stitch PNGs into video
    call([
        "ffmpeg",
        "-y",
        "-framerate", "20",
        "-i", str(vis_path / "iter_%02d_from_svg.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ], stdout=DEVNULL, stderr=DEVNULL)

    if delete_images:
        shutil.rmtree(vis_path)

