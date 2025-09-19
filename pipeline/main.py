from pathlib import Path
from prep import run as run_prep
from fit import batch as run_fit
from visualization import run as run_vis
from fit import sh

import ipdb

exp_path = sh.exp_path
raw_path = Path("./raw").resolve()
data_path = Path("./data").resolve()

def main():

    run_prep(raw_path, data_path)

    # ipdb.set_trace()

    run_fit(data_path, sh.w)


    run_vis(sh.exp_path, sh.exp_path / f"gallery_{sh.w}")

if __name__ == "__main__":
    main()