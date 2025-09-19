import numpy as np
from pathlib import Path
import torch

class Share:
    def __init__(self):
        pass


sh = Share()

sh.epoch = 200
sh.num_samples = 32
sh.color_guess = (1, 1, 1, 1.)
sh.smooth_weight = 10
sh.exp_name = "pipeline"

sh.w = 64

sh.exp_path = Path("./exp/").resolve()


