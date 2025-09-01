from pathlib import Path

# subfolder-level

def run(subfold_path):

    
    pass


if __name__ == "__main__":
    run(subfold = Path(""),
         )
    

def batch(fold):
    
    # one-level only
    for subfold in fold.glob("*/"):
        run(subfold)

if __name__ == "__main__":
    
    fold = Path("")
    batch(fold)