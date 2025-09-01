def batch(fold, legal_suffix):
    
    # one-level only
    for subfold in fold.glob("*/"):
        for img in subfold.glob("*/"):
            name, suffixs = img.stem.split("@")[0], img.stem.split("@")[1:]
            for suffix in suffixs:
                if suffix not in legal_suffix:
                    print(f"detect illegal suffix in {img}")
            img.rename(img.with_stem(name))
            # print(img)
            
            
        
            