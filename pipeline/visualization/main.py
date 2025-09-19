import os
import subprocess
from pylatex import *
from pathlib import Path
from tqdm import tqdm
import ipdb
import shutil

from fit import sh

def get_max_iter(sub_path):
    import re
    vis_path = sub_path / "vis"
    max_num = -1

    # regex to match files like iter_123
    pattern = re.compile(r"iter_(\d+)")

    for fname in os.listdir(vis_path):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num if max_num != -1 else None

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


class MyDocument3Column(Document):
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

        self.append(NoEscape(r"\maketitle"))

        self.append(NoEscape(r"""
        \section*{Illustration Guide}
        \begin{itemize}
            \item \textbf{Target}: Given raster image that we want to approximate with curves.
            \item \textbf{Init Vector}: Initial vertices and edges.
            \item \textbf{Init Render}: Rasterized image from init vector.
            \item \textbf{Vector Result}: Final optimized vertices and edges.
            \item \textbf{Render Result}: Rasterized image from vector result.
            \item \textbf{Loss Curve}: Training loss evolution over iterations.
        \end{itemize}
        """))

        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path, doc):

        for sub_path in tqdm(
            sorted([p for p in image_path.iterdir() if p.is_dir()]),
            desc="visualizing"
        ):

            # sub_path = sub_path / str(16)
            if not sub_path.exists() or (sub_path / "render.png").exists() is False:
                continue

            image_keys = ['target', "init vector", "init render", "vector result", "render result",\
                           "loss curve", "segmentation", "2", "3", "4", "5", "6"]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'init_vec.png',
                    'init_render.png',
                    'vec.png',
                    'render.png',
                    'loss.png',
                    'contour.png',
                    'contour.png',
                    'contour.png',
                    'contour.png',
                    'contour.png',
                    'contour.png',

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
                images.add_caption(sub_path.name)

                self.count += 1
                if self.count % 3 == 0:
                    self.append(NoEscape(r'\clearpage'))

class MyDocument4Column(Document):
    def __init__(self, output_path):
        super().__init__(output_path, inputenc=None)

        # Metadata
        self.preamble.extend([
            Command('title', 'Image Gallery'),
            Command('author', 'Ziyu Sun'),
        ])

        # Required packages
        self.packages.update([
            Package('graphicx'),
            Package('subcaption'),
            Package('float'),
            Package('indentfirst'),
        ])

        # Title
        self.append(NoEscape(r"\maketitle"))

        # Intro guide
        self.append(NoEscape(r"""
        \section*{Illustration Guide}
        For ablation purpose, I replace previous smooth loss(which counts for both distance and direction) with only distance. I then add the curvature loss with a large weight (Now we only have image loss, distance-smooth loss and curvature loss).
                                            
        There are 200 training iterations in total. The first 100 iterations are without curvature loss, and the last 100 iterations are with curvature loss. 
                             
        The final results may not look very good, but it demonstrates that the current curvature loss works. For better fitting we can finetune the weights maybe in the future.
            \begin{itemize}
            \item \textbf{Target}: Given raster image that we want to approximate with curves.
            \item \textbf{Contour}: Contour extracted from target image with grey value threasholding.
            \item \textbf{Init Vector}: Initial vertices and edges from contour.
            \item \textbf{Init Render}: Rasterized image from init vector.
            \item \textbf{vector 100th}: Vertices and edges from the 100th iteration without curvature loss.
            \item \textbf{render 100th}: Rasterized image from the 100th iteration without curvature loss.
            \item \textbf{Vector Result}: Final optimized vertices and edges after 200 iterations.
            \item \textbf{Render Result}: Rasterized image from vector result.
            \item \textbf{Loss Curve}: Training loss evolution over iterations.
        \end{itemize}
        """))

        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path: Path, doc: Document):

        for sub_path in tqdm(
            sorted([p for p in image_path.iterdir() if p.is_dir()]),
            desc="visualizing"
        ):

            if not sub_path.exists() or (sub_path / "render.png").exists() is False:
                continue

            image_keys = [
                'target', "contour", "init vector", "init render", "vector 100th", "render 100th", "vector result", "render result",
                "loss curve",
            ]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'contour.png',
                    'init_vec.png',
                    'init_render.png',
                    'vec_first_stage.png',
                    'render_first_stage.png',
                    'vec.png',
                    'render.png',
                    'loss.png',
                ])
            }

            # 4-column figure
            with doc.create(Figure(position="H")) as images:
                for i, key in enumerate(image_keys):
                    # first page
                    with doc.create(
                        SubFigure(position="b", width=NoEscape(r"0.24\linewidth"))
                    ) as subfig:

                        img_path = image_paths[key]

                        if img_path.suffix == '.svg':
                            # Convert SVG → PDF if needed
                            pdf_path = convert_svg_to_pdf(img_path)
                            subfig.add_image(str(pdf_path), width=NoEscape(r"\linewidth"))
                        else:
                            subfig.add_image(str(img_path), width=NoEscape(r"\linewidth"))

                        subfig.add_caption(key)

                    # Break line after 4 subfigures
                    if (i + 1) % 4 == 0:
                        doc.append(NoEscape(r"\par\vspace{1em}"))
                
                # next page
            doc.append(NoEscape(r'\newpage'))

                # -------- Second page: iter_xx + render_xx timeline --------
            with doc.create(Figure(position="H")) as timeline:
                count_in_row = 0
                # ipdb.set_trace()
                num_images = get_max_iter(sub_path)
                for i in range(0, num_images, int((num_images+1)/10)):  # 09, 19, 29, ..., 99
                    idx = f"{i:03d}"
                    iter_img = sub_path / "vis" / f"iter_{idx}.png"
                    render_img = sub_path / "vis" / f"render_iter_{idx}.png"
                    # ipdb.set_trace()
                    if not (iter_img.exists() and render_img.exists()):
                        continue

                    # iter_xx
                    with doc.create(SubFigure(position="b", width=NoEscape(r"0.24\linewidth"))) as subfig:
                        subfig.add_image(str(iter_img), width=NoEscape(r"\linewidth"))
                        subfig.append(NoEscape(r"\caption*{%sth iter (vector)}" % idx))
                    count_in_row += 1

                    # render_xx
                    with doc.create(SubFigure(position="b", width=NoEscape(r"0.24\linewidth"))) as subfig:
                        subfig.add_image(str(render_img), width=NoEscape(r"\linewidth"))
                        subfig.append(NoEscape(r"\caption*{%sth iter (render)}" % idx))
                    count_in_row += 1

                    # break after 4 images (two pairs)
                    if count_in_row == 4:
                        doc.append(NoEscape(r"\par\vspace{1em}"))
                        count_in_row = 0


                # timeline.add_caption(f"Iteration timeline for {sub_path.name}")

                

                # Caption for the whole figure group
                # images.add_caption(sub_path.name)

                # Count and insert page breaks if needed
                self.count += 1
                if self.count % 4 == 0:  # adjust to 4 if you want page breaks aligned
                    self.append(NoEscape(r'\clearpage'))



def run_latex(image_path, output_path, delete_vis=False):
    doc = MyDocument4Column(output_path)
    doc.fill_document(image_path, doc)
    doc.generate_pdf(
        clean_tex=False,
        compiler="pdflatex",
        compiler_args=["-interaction=nonstopmode"]
    )



if __name__ == "__main__":

    exp_path = Path(r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\09-14\22-13-56")
    output_path = exp_path / "res"
    image_path = exp_path
    run_latex(image_path, output_path, delete_vis=False)
