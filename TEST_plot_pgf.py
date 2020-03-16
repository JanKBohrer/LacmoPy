import matplotlib as mpl
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

import matplotlib.pyplot as plt
import numpy as np

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",         # use Xelatex which is TTF font aware
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Ubuntu",             # use 'Ubuntu' as the standard font
    "font.sans-serif": [],
    "font.monospace": "Ubuntu Mono",    # use Ubuntu mono if we have mono
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
#    "text.latex.unicode": True,
    "pgf.preamble": [
            r'\usepackage[T1]{fontenc}',
#        r'\usepackage{unicode-math}'
    ]
#    "pgf.preamble": [
#        r'\usepackage{fontspec}',
#        r'\setmainfont{Ubuntu}',
#        r'\setmonofont{Ubuntu Mono}',
#        r'\usepackage{unicode-math}'
#        r'\setmathfont{Ubuntu}'
#    ]
}

mpl.rcParams.update(pgf_with_latex)


x = np.arange(100)
fig, ax = plt.subplots()

ax.plot(x, np.sin(x))
#ax.set_title(r"1000.222 \si{\micro\meter}")
ax.set_title(r"1000.222")
fig.savefig("test1000_siunitx_error.pdf")