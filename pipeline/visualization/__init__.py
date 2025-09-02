from .main import run_latex
from .utils import add_suffix

def run(data_path, gallery_path):

    # add suffix @subfold for svg renderng
    # add_suffix(data_path)

    # raster_svg(data_path)

    run_latex(data_path, gallery_path)