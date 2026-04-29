# Frequently Used Commands

## Virtual Environment

All Python work in this project uses `retrieval_exp_env`.

```bash
# Activate
source /Users/saikrishnab/LearningAndDevelopment/virtualenvs/retrieval_exp_env/bin/activate

# Deactivate
deactivate

# Check active env
which python3   # should show .../retrieval_exp_env/bin/python3
```

---

## Figures — Add / Convert Images for report.tex

Images live in `docs/figures/`. pdflatex needs RGB PNG or JPG (not 16-bit or palette modes).
Always convert downloaded images before adding them to the report.

```bash
source /Users/saikrishnab/LearningAndDevelopment/virtualenvs/retrieval_exp_env/bin/activate

python3 - <<'EOF'
from PIL import Image
img = Image.open("docs/figures/your_image.png")
print(img.size, img.mode)          # verify
img.convert("RGB").save("docs/figures/your_image_rgb.png")
EOF
```

To convert an SVG to PDF (needed for pdflatex):
```bash
source /Users/saikrishnab/LearningAndDevelopment/virtualenvs/retrieval_exp_env/bin/activate
python3 -c "import cairosvg; cairosvg.svg2pdf(url='input.svg', write_to='output.pdf')"
```

`\graphicspath{{figures/}}` is set in `report.tex` — use bare filenames in `\includegraphics{}`.

---

## LaTeX — Compile report.tex to PDF

```bash
cd /Users/saikrishnab/LearningAndDevelopment/retrieval_evolution_study/docs
pdflatex report.tex
pdflatex report.tex   # run twice to resolve cross-references (TOC, \ref{}, citations)
```

Run twice every time — first pass compiles content, second pass fixes page numbers and `??` references.

### If you added bibliography entries (`\cite{}`)

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

### Open the output

```bash
open report.pdf
```
