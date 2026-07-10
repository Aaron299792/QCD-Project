import numpy as np
from scipy.special import loggamma
from scipy.integrate import quad
from scipy.optimize import fsolve

class Conversion:
    def __init__(self):
        self.hbarc= 0.1973
        self.fm_to_GeVm1 = 1.0 /self.hbarc
        self.fm2_to_GeVm2 = self.fm_to_GeVm1 * self.fm_to_GeVm1
        self.GeV_to_fmm1 = 1.0 / self.hbarc
        self.GeV2_to_fmm2 = self.GeV_to_fmm1 * self.GeV_to_fmm1
        self.GeVm1_to_fm = self.hbarc
        self.GeVm2_to_fm2 = self.GeVm1_to_fm * self.GeVm1_to_fm
        self.fmm1_to_GeV = self.hbarc
        self.fmm2_to_GeV2 = self.fmm1_to_GeV * self.fmm1_to_GeV
        self.mb_to_fm2 = 0.1
        self.fm2_to_mb = 1.0 / self.mb_to_fm2

class Params:
    def __init__(self, Q02, NC, alpha_S, BG, m, root_sNN, sigmaNN, la, delt=1.0, K = 2.0):
        self.Q02 = Q02
        self.CF = NC * NC - 1 / (2.0 * NC)
        self.alpha_S = alpha_S
        self.BG = BG
        self.s0 = 2.0 * np.pi * BG
        self.m = m
        self.sqsNN = root_sNN
        self.la = la
        self.delt = delt
        self.sigmaNN = sigmaNN
        self.dmax = np.sqrt(0.1 * sigmaNN / np.pi)
        self.K = K

    def print(self):
        print("-"*35)
        print("Parameters: ")
        print("-"*35)
        print(f"Q_0^2 : {self.Q02} GeV")
        print(f"C_F : {self.CF:.3f}")
        print(f"Alpha_S : {self.alpha_S}")
        print(f"B_G : {self.BG} fm^2")
        print(f"sigma_0 : {self.s0:.3f} fm^2")
        print(f"mass : {self.m} GeV")
        print(f"sqrt(s_NN) : {self.sqsNN} GeV")
        print(f"sigmaNN : {self.sigmaNN} mb")
        print(f"Dmax : {self.dmax} fm")
        print(f"lambda : {self.la}")
        print(f"delta : {self.delt}")
        print(f"K_g : {self.K}")
        print("-"*35)

class Atom:
    def __init__(self, A, R, a, name):
        self.A = A
        self.R = R
        self.a = a
        self.rmax = R + 6.25 * a
        self.name = name

    def print(self):
        print("-"*41)
        print(f"Atom: {self.name}")
        print("-"*41)
        print(f"Atomic Number : {self.A}")
        print(f"Nuclear Radius : {self.R} fm")
        print(f"Nuclear diffusivity constant : {self.a} fm")
        print(f"Maximum radius : {self.rmax:.2f} fm")
        print("-"*41)

class Utils(Atom):
    def __init__(self, A, R, a, name, seed = None):
        super().__init__(A, R, a, name)
        self.norm = self.A / (4.0 * np.pi * quad(self.r2WS, 0, self.rmax)[0])
        self.r_optim, self.w_max, self.eta = self._optim()
        self.rng = np.random.default_rng(seed = seed)

    def woods_saxon(self, r):
        return 1.0 / (1.0 + np.exp((r - self.R) / self.a))

    def r2WS(self, r):
        return r**2 / (1.0 + np.exp((r - self.R) / self.a))

    def _optim(self):
        f = lambda r : 2.0 - r * (1.0 - self.woods_saxon(r)) / self.a
        r_optim = fsolve(f, self.R)[0]
        w_max = self.r2WS(r_optim)
        eta = quad(self.r2WS, 0, self.rmax)[0] / (self.rmax * w_max)
        return r_optim, w_max, eta

    def sample_radii_rejection(self, n):
        n_prop = int(1.5 * n / self.eta)
        r = self.rng.uniform(0.0, self.rmax, n_prop)
        w = self.r2WS(r)
        u = self.rng.uniform(0.0, 1.0, n_prop)
        accepted = r[u < w / self.w_max]
        return accepted[:n]

if __name__=="__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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

    n_samples = 10000
    utils = Utils(A = 208, R = 6.62, a = 0.546, name="Lead (Pb)")
    r_samples = utils.sample_radii_rejection(n_samples)

    r = np.linspace(0, utils.rmax, 600)
    w = utils.r2WS(r)
    pdf = w * utils.norm * 4.0 * np.pi / utils.A

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r_samples, bins=60, density=True, alpha=0.4,
            color="darkgreen", label=f"MC (N={n_samples} nucleons)")
    ax.plot(r, pdf, "k--", lw=2,
            label=r"$ r^2 \rho(r)$")
    ax.set_xlabel(r"\textbf{Radius} r [fm]")
    ax.set_ylabel(r"\textbf{Probability distribution}")
    ax.legend(loc=2, framealpha=0.2)
    ax.set_ylim([0, 0.3])
    fig.tight_layout()
    plt.savefig("../../plots/radii_sampling.pdf", format='pdf', dpi=300, bbox_inches='tight')





