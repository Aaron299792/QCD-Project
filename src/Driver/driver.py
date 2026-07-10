import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": True,
    "lines.antialiased": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.linewidth" : 2.0,
    "xtick.major.width" : 2.0,
    "ytick.major.width" : 2.0
})


sys.path.append(os.path.abspath('../modules'))
from utils import Utils, Params
from thickness import sample_nucleus

def sample_AB(b, utils):
    nucleus = sample_nucleus(utils)[:, :2]
    xA = nucleus - 0.5 * b
    xB = nucleus + 0.5 * b
    return xA, xB

def interaction_matrix(posA, posB, pars):
    diff = posA[:, None, :] - posB[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    return dist < pars.dmax

b = np.array([4.0, 0.0])
n_samples = 500
utils = Utils(A=208, R=6.62, a=0.546, name="Lead (Pb)", seed=42)
params = Params(Q02=0.152, NC=3, alpha_S=0.3, BG=0.156, m=0.1, root_sNN=2760, sigmaNN=64, la=0.215, delt=5.6)
xA, xB = sample_AB(b, utils)
interacts = interaction_matrix(xA, xB, params)
Ncoll = interacts.sum()
nucleon_A = interacts.any(axis=1)
nucleon_B = interacts.any(axis=0)
Npart = nucleon_A.sum() + nucleon_B.sum()

indexA, indexB = np.where(interacts)

fig, ax = plt.subplots()
circA = plt.Circle(-0.5 * b,utils.R, edgecolor='#000000', facecolor='none', linestyle='--')
circB = plt.Circle(0.5*b ,utils.R, edgecolor='#000000', facecolor='none', linestyle='--')
ax.scatter(xA[~nucleon_A, 0], xA[~nucleon_A, 1], s=25,facecolors='none', edgecolors='#0A36AF')
ax.scatter(xA[nucleon_A, 0], xA[nucleon_A, 1], s=35, color="#0A36AF")
ax.scatter(xB[~nucleon_B, 0], xB[~nucleon_B, 1], s=25,facecolors='none', edgecolors='#800000')
ax.scatter(xB[nucleon_B, 0], xB[nucleon_B, 1], s=35, color="#800000")
ax.set_ylabel('Y [fm]')
ax.set_xlabel('X [fm]')
ax.add_patch(circA)
ax.add_patch(circB)
ax.text(-10.0, 6.5, r'$^{208}$Pb - $^{208}$Pb')
plt.tight_layout()
plt.savefig("../../plots/PbPb_interact.pdf", format='pdf', dpi=300)
plt.close()


