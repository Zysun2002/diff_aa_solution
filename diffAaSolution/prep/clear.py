import shutil
from pathlib import Path

def run(fold, keep_list):
    keep_files = set(keep_list)  # faster lookups

    for item in fold.iterdir():
        if item.is_file():
            if item.name not in keep_files:
                item.unlink()  # delete file
        elif item.is_dir():
            shutil.rmtree(item)  # delete folder

    
def batch(fold_path, keep_list):
    # folder-level

    # one-level only
    for subfold_path in fold_path.glob("*/"):
        run(subfold_path, keep_list)

if __name__ == "__main__":
    
    fold_path = Path("/workspace/diffvg/diffAaSolution/data")
    keep_list = ["aa_16.png", "aa_32.png", "aa_64.png", "aa_128.png"]
    batch(fold_path, keep_list)