from pathlib import Path
import shutil

# fold-level

def run(fold):

    # raname 
    mapping = {
        "bridge/bridge_32.png": "bridge/32_image.png",
        "bridge/bridge_64.png": "bridge/64_image.png",
        "bridge/bridge_128.png": "bridge/128_image.png",
        "bridge/bridge_256.png": "bridge/256_image.png",
        "bridge45/bridgelow45_32.png": "bridge45/32_image.png",
        "bridge45/bridgelow45_64.png": "bridge45/64_image.png",
        "bridge45/bridge45_128.png": "bridge45/128_image.png",
        "bridge45/bridge45_256.png": "bridge45/256_image.png",
        "circle/circle_32.png": "circle/32_image.png",
        "circle/circle_64.png": "circle/64_image.png",
        "circle/circle_128.png": "circle/128_image.png",
        "circle/circle_256.png": "circle/256_image.png",
        "circlesmall/circlesmall_32.png": "circlesmall/32_image.png",
        "circlesmall/circlesmall_64.png": "circlesmall/64_image.png",
        "circlesmall/circlesmall_128.png": "circlesmall/128_image.png",
        "circlesmall/circlesmall_256.png": "circlesmall/256_image.png",
        "curve/curve_32.png": "curve/32_image.png",
        "curve/curve_64.png": "curve/64_image.png",
        "curve/curve_128.png": "curve/128_image.png",
        "curve/curve_256.png": "curve/256_image.png",
        "ellipse/ellipse_32.png": "ellipse/32_image.png",
        "ellipse/ellipse_64.png": "ellipse/64_image.png",
        "ellipse/ellipse_128.png": "ellipse/128_image.png",
        "ellipse/ellipse_256.png": "ellipse/256_image.png",
        "ellipse45/ellipse45_32.png": "ellipse45/32_image.png",
        "ellipse45/ellipse45_64.png": "ellipse45/64_image.png",
        "ellipse45/ellipse45_128.png": "ellipse45/128_image.png",
        "ellipse45/ellipse45_256.png": "ellipse45/256_image.png",
        "l/l_32.png": "l/32_image.png",
        "l/l_64.png": "l/64_image.png",
        "l/l_128.png": "l/128_image.png",
        "l/l_256.png": "l/256_image.png",        
        "l45/l45_32.png": "l45/32_image.png",
        "l45/l45_64.png": "l45/64_image.png",
        "l45/l45_128.png": "l45/128_image.png",
        "l45/l45_256.png": "l45/256_image.png",
        "sfont2/sfont2_32.png": "sfont2/32_image.png",
        "sfont2/sfont2_64.png": "sfont2/64_image.png",
        "sfont2/sfont2_128.png": "sfont2/128_image.png",
        "sfont2/sfont2_256.png": "sfont2/256_image.png",
        "square/square_32.png": "square/32_image.png",
        "square/square_64.png": "square/64_image.png",
        "square/square_128.png": "square/128_image.png",
        "square/square_256.png": "square/256_image.png",
        "square30/square30_32.png": "square30/32_image.png",
        "square30/square30_64.png": "square30/64_image.png",
        "square30/square30_128.png": "square30/128_image.png",
        "square30/square30_256.png": "square30/256_image.png",
        "square45/square45_32.png": "square45/32_image.png",
        "square45/square45_64.png": "square45/64_image.png",
        "square45/square45_128.png": "square45/128_image.png",
        "square45/square45_256.png": "square45/256_image.png",
        "squaresmall/squaresmall_32.png": "squaresmall/32_image.png",
        "squaresmall/squaresmall_64.png": "squaresmall/64_image.png",
        "squaresmall/squaresmall_128.png": "squaresmall/128_image.png",
        "squaresmall/squaresmall_256.png": "squaresmall/256_image.png",
        "squaresmall45/squaresmall45_32.png": "squaresmall45/32_image.png",
        "squaresmall45/squaresmall45_64.png": "squaresmall45/64_image.png",
        "squaresmall45/squaresmall45_128.png": "squaresmall45/128_image.png",
        "squaresmall45/squaresmall45_256.png": "squaresmall45/256_image.png",
        "tie/tie_32.png": "tie/32_image.png",
        "tie/tie_64.png": "tie/64_image.png",
        "tie/tie_128.png": "tie/128_image.png",
        "tie/tie_256.png": "tie/256_image.png",
    }


    for old_name, new_name in mapping.items():
        old_path = fold / old_name
        new_path = fold / new_name

        if old_path.exists():
            shutil.copy2(old_path, new_path) 



if __name__ == "__main__":
    run(fold = Path("/workspace/diffvg/diffAaSolution/data"))