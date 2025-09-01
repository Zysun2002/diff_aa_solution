from pathlib import Path
from .main import run


# folder-level
def batch(fold):
    
    # one-level only
    for subfold in fold.glob("*/"):
        run(subfold)

if __name__ == "__main__":
    
    fold = Path("")
    batch(fold)