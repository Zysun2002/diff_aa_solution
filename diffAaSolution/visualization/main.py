import os
import ipdb
from pylatex import * 
from pathlib import Path

from fit import sh

class MyDocument(Document):
    def __init__(self, output_path):
        super().__init__(output_path, inputenc=None)

        self.preamble.extend([
            Command('title', 'Image Gallery'),
            Command('author', 'Ziyu Sun'),
        ])

        self.packages.update([
            Package('graphicx'),
            Package('subcaption'), 
            Package('float')
        ])
        
        self.packages.append(Command('usepackage', 'svg'))

        self.append(NoEscape(r"\maketitle"))
        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path, doc):

        for sub_path in sorted([p for p in image_path.iterdir() if p.is_dir()]):


            sub_path = sub_path / str(sh.w)
            if not sub_path.exists():
                continue

            image_keys = ['target', "res-raster", "res-svg"]
            # image_keys = ['anti_32', "aliased_64"]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'res.png',
                    'res.svg'
                ])
            }

            with doc.create(Figure(position="H")) as images:
                for i, key in enumerate(image_keys):
                    with doc.create(
                        SubFigure(position="b", width=NoEscape(r"0.32\linewidth"))
                    ) as subfig:
                        # ipdb.set_trace()
                        if image_paths[key].suffix == '.svg':

                            # add suffix
                            svg_path = image_paths[key].with_stem(image_paths[key].stem + "@" + sub_path.parent.name)
                            subfig.append(NoEscape(
                                rf"\includesvg[width=\linewidth]{{{svg_path}}}"
                            ))
                        else:
                            # Normal raster image
                            subfig.add_image(str(image_paths[key]), width=NoEscape(r"\linewidth"))
            
                        # ipdb.set_trace()
                        subfig.add_caption(key)
                        
                    if (i+1) % 3 == 0:
                        doc.append(NoEscape(r"\par\vspace{1em}"))

                name = sub_path.name
                images.add_caption(name[4:])
                
                self.count += 1
                if self.count % 3 == 0:
                    self.append(NoEscape(r'\clearpage'))


def run_latex(image_path, output_path):
    doc = MyDocument(output_path)
    doc.fill_document(image_path, doc)
    doc.generate_pdf(
        clean_tex=False, 
        compiler="pdflatex", 
        compiler_args=["-shell-escape"]
    )
       

if __name__ == "__main__":
    output_path = Path("/workspace/diffvg/diffAaSolution/data/gallery")

    image_path = Path("/workspace/diffvg/diffAaSolution/data/")

    run_latex(image_path, output_path)
