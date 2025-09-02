from pathlib import Path
import shutil
from .clean import run as run_clean
from .clear import batch as run_clear
from .downsample import batch as run_downsample
from .solidify import batch as run_solidify
from .unify import batch as run_unify


def run(raw_path, fold):
    # delete label_ui.py
    if fold.exists():
        shutil.rmtree(fold)

    shutil.copytree(raw_path, fold)

    ui_path = fold / "label_ui.py"
    if ui_path.exists():
        ui_path.unlink()

    # change some of the irregular names
    run_clean(fold) # 

    # downsample to get anti-alaised images
    run_downsample(fold)

    # delete irrelevant files
    run_clear(fold, keep_list=["aa_16.png", "aa_32.png", "aa_64.png", "aa_128.png"])
    
    # convert images with transparent background to balck-white images
    run_solidify(fold)
    #! this will introduce subfix of solid and invert

    run_unify(fold, legal_suffix=["solified", "inverted"])






if __name__ == "__main__":
    fold = Path("/workspace/diffvg/diffAaSolution/data")
    run(fold)