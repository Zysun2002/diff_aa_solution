from pathlib import Path
from prep import run as run_prep
from fit import batch as run_fit
from visualization import run as run_vis
from fit import sh

import ipdb

raw_path = Path("./raw").resolve()
exp_path = Path("./exp/try_init").resolve()
data_path = Path("./data").resolve()

def main():
    run_prep(raw_path, data_path)

    sh.w = 16
    run_fit(data_path, 16)

    run_vis(data_path, data_path / f"gallery_{sh.w}")

if __name__ == "__main__":
    main()