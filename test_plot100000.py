import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(plt.rcParamsDefault)
mpl.use("pgf")

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict, pgf_dictX
plt.rcParams.update(pgf_dict)

#%%

x = np.linspace(0,6.3,100)
fig, ax = plt.subplots()

ax.plot(x, np.sin(x))
ax.set_title(r"1000.222 \si{\micro\meter}")
#ax.set_title(r"1000.222")
fig.savefig("test1000_siunitx_error.pdf")

