import os
import subprocess
from pylatex import *
from pathlib import Path

from fit import sh


def convert_svg_to_pdf(svg_path: Path) -> Path:
    """
    Convert an SVG file to PDF using Inkscape CLI.
    Returns the path to the generated PDF.
    """
    pdf_path = svg_path.with_suffix(".pdf")

    # Run Inkscape conversion (must be in PATH)
    subprocess.run([
        "inkscape",
        str(svg_path),
        "--export-type=pdf",
        f"--export-filename={pdf_path}"
    ], check=True)

    return pdf_path


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

        # No more \includesvg → we don’t need svg package
        self.append(NoEscape(r"\maketitle"))
        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path, doc):

        for sub_path in sorted([p for p in image_path.iterdir() if p.is_dir()]):

            sub_path = sub_path / str(16)
            if not sub_path.exists():
                continue

            image_keys = ['target', "res-raster", "res-svg", "init guess"]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'res.png',
                    'res.svg',
                    'init.svg'
                ])
            }

            with doc.create(Figure(position="H")) as images:
                for i, key in enumerate(image_keys):
                    with doc.create(
                        SubFigure(position="b", width=NoEscape(r"0.32\linewidth"))
                    ) as subfig:

                        img_path = image_paths[key]

                        if img_path.suffix == '.svg':
                            # Convert SVG → PDF once
                            pdf_path = convert_svg_to_pdf(img_path)
                            subfig.add_image(str(pdf_path), width=NoEscape(r"\linewidth"))
                        else:
                            subfig.add_image(str(img_path), width=NoEscape(r"\linewidth"))

                        subfig.add_caption(key)

                    if (i + 1) % 3 == 0:
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
        compiler_args=["-interaction=nonstopmode"]
    )


if __name__ == "__main__":
    output_path = Path("E:\Ziyu\workspace\diff_aa_solution\pipeline\data")
    image_path = Path("E:\Ziyu\workspace\diff_aa_solution\pipeline\data")
    run_latex(image_path, output_path)
