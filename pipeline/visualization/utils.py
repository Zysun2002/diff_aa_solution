from pathlib import Path
import shutil

from fit import sh

def add_suffix(fold: Path):

    for subfold in sorted(p for p in fold.iterdir() if p.is_dir()):
        svg_dir = subfold / str(sh.w)
        if not svg_dir.exists():
            continue

        for svg in svg_dir.glob("*.svg"):
            
            if svg.stem.endswith(f"@{subfold.name}"):
                continue

            new_name = f"{svg.stem}@{subfold.name}{svg.suffix}"
            new_path = svg_dir / new_name

            shutil.copy(svg, new_path)   # or use svg.rename(new_path) to move

def raster_svg(fold: Path):
    import subprocess
    from multiprocessing import Pool

    def worker(svg_path):
        png_path = svg_path.with_suffix(".png")
        cmd = ["rsvg-convert", "-h", "256", "-w", "256", "-o", str(png_path), str(svg_path)]
        subprocess.run(cmd, check=True)

    svg_files = list(fold.rglob("*.svg"))
    with Pool() as pool:
        pool.map(worker, svg_files)

# Example usage
if __name__ == "__main__":
    fold = Path("/workspace/diffvg/diffAaSolution/data")         # your input root
    add_suffix(fold)
