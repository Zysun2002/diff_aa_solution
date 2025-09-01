from PIL import Image, ImageOps
from pathlib import Path


def convert_transparent_to_black_white(img_path, out_path):
    # object-level
        
    img = Image.open(img_path).convert("RGBA")

    # Extract alpha channel (transparency)
    alpha = img.split()[3]

    # Use alpha directly: object = white (opaque), background = black (transparent)
    bw_image = alpha.convert("L")  # L = grayscale

    bw_image.save(out_path)

def invert_black_white(img_path, out_path):
        # Open your black/white or grayscale image
        img = Image.open(img_path).convert("L")  # "L" = grayscale

        # Invert black <-> white
        inverted = ImageOps.invert(img)

        # Save the result
        inverted.save(out_path)


def batch(fold):
    process_list = [
        fold / "bridge/aa_16.png",
        fold / "bridge/aa_32.png",
        fold / "bridge45/aa_32.png",
        fold / "bridge45/aa_64.png",
        fold / "bridge45/aa_128.png",
        fold / "circle/aa_64.png",
        fold / "circle/aa_128.png",
        fold / "circlesmall/aa_16.png",
        fold / "circlesmall/aa_32.png",
        fold / "circlesmall/aa_64.png",
        fold / "circlesmall/aa_128.png",
        fold / "curve/aa_16.png",
        fold / "curve/aa_64.png",
        fold / "curve/aa_128.png",
        fold / "ellipse/aa_16.png",
        fold / "ellipse/aa_32.png",
        fold / "ellipse/aa_64.png",
        fold / "ellipse/aa_128.png",
        fold / "ellipse45/aa_16.png",
        fold / "ellipse45/aa_32.png",
        fold / "ellipse45/aa_64.png",
        fold / "ellipse45/aa_128.png",
        fold / "l/aa_16.png",
        fold / "l/aa_32.png",
        fold / "l/aa_64.png",
        fold / "l/aa_128.png",
        fold / "l45/aa_16.png",
        fold / "l45/aa_32.png",
        fold / "l45/aa_64.png",
        fold / "l45/aa_128.png",
        fold / "sfont2/aa_16.png",
        fold / "sfont2/aa_32.png",
        fold / "sfont2/aa_64.png",
        fold / "sfont2/aa_128.png",        
        fold / "square/aa_16.png",
        fold / "square/aa_32.png",
        fold / "square/aa_64.png",
        fold / "square30/aa_16.png",
        fold / "square30/aa_32.png",
        fold / "square30/aa_64.png",
        fold / "square30/aa_128.png",  
        fold / "square45/aa_16.png",
        fold / "square45/aa_32.png",
        fold / "square45/aa_64.png",
        fold / "square45/aa_128.png",
        fold / "squaresmall/aa_16.png",
        fold / "squaresmall/aa_32.png",
        fold / "squaresmall/aa_64.png",
        fold / "squaresmall/aa_128.png",  
        fold / "squaresmall45/aa_16.png",
        fold / "squaresmall45/aa_32.png",
        fold / "squaresmall45/aa_64.png",
        fold / "squaresmall45/aa_128.png",
        fold / "tie/aa_16.png",
        fold / "tie/aa_32.png",
        fold / "tie/aa_64.png",
        fold / "tie/aa_128.png",  
    ]

    for img_path in process_list:
        out_path = img_path.with_stem(img_path.stem + "@solified")
        if not out_path.exists():
            convert_transparent_to_black_white(img_path, out_path)
            img_path.unlink()


    invert_list = [
        fold / "circle/aa_16.png",
        fold / "circle/aa_32.png",
        fold / "square/aa_128.png",
    ]

    for img_path in invert_list:
        out_path = img_path.with_stem(img_path.stem + "@inverted")
        if not out_path.exists():
            invert_black_white(img_path, out_path)
            img_path.unlink()





if __name__ == "__main__":
    img_path = Path("/workspace/diffvg/diffAaSolution/data") / "circle/aa_32.png"
    out_path = Path("/workspace/diffvg/diffAaSolution/data") / "circle/aa_32.png"
    invert_black_white(img_path, out_path)