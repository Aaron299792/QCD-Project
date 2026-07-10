import numpy as np
from scipy.integrate import simpson
from utils import Utils, Params

def sample_nucleus(utils):
    r = utils.sample_radii_rejection(utils.A)
    costheta = utils.rng.uniform(-1, 1, utils.A)
    theta = np.arccos(costheta)
    phi = utils.rng.uniform(0, 2.0 * np.pi, utils.A)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)

def thickness(X, Y, positions_xy, pars):
    T = np.zeros_like(X)
    norm = 1.0 / (2.0 * np.pi * pars.BG)
    for (xi, yi) in positions_xy:
        T += norm * np.exp(-((X - xi)**2 + (Y - yi)**2) / (2.0 * pars.BG))
    T[T < 1.0e-12] = 0.0
    return T

def thickness_byintegral(s_array, zmax, utils):
    z = np.linspace(-zmax, zmax, 4000)
    T = np.zeros_like(s_array)
    for i, s in enumerate(s_array):
        r = np.sqrt(s**2 + z**2)
        rho = utils.woods_saxon(r)
        T[i] = simpson(rho, z)
    return T

if __name__=="__main__":
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

    n_events = 1000
    xmax = 12.0
    ymax = 12.0
    zmax = 12.0
    #smax = np.sqrt(xmax**2 + ymax**2)
    NX, NY =241, 241

    utils = Utils(A=208, R=6.62, a=0.546, name="Lead (Pb)", seed=42)
    params = Params(Q02=0.152, NC=3, alpha_S=0.3, BG=0.156, m=0.1, root_sNN=2760, sigmaNN=64, la=0.215, delt=5.6)
    x = np.linspace(-xmax, xmax, 241)
    y = np.linspace(-ymax, ymax, 241)
    iy0 = np.argmin(np.abs(y))
    s = np.sqrt(x**2 + y**2)
    X, Y = np.meshgrid(x,y, indexing="ij")
    pos_sample = sample_nucleus(utils)
    T_sample = thickness(X, Y, pos_sample[:, :2], params)

    T_avg = np.zeros(shape=(NX, NY))
    for _ in range(n_events):
        pos = sample_nucleus(utils)
        T_avg += thickness(X, Y, pos[:, :2], params)
    T_avg /= n_events

    rho0 = utils.norm
    T_int = rho0 * thickness_byintegral(s_array=s, zmax=zmax, utils=utils)

    x_p = x >= 0

    im = plt.contourf(X,Y,T_sample,levels=17, cmap='inferno')
    cbar = plt.colorbar(im)
    cbar.set_label(r'\textbf{Thickness} $T(\textbf{x}_\perp)$ [fm$^{-2}$]', rotation=270, labelpad=20)
    plt.xlabel(r'X [fm]')
    plt.ylabel(r'Y [fm]')
    plt.tight_layout()
    plt.savefig('../../plots/thickness_v2.pdf', format='pdf', dpi=300)
    plt.close()

    plt.plot(s, T_int, color='k', lw=2.0,label='Analytical')
    plt.plot(x[x_p], T_avg[x_p, iy0], color="#DD571C", linestyle='--', lw=3.0, label=r' MC $\langle T(\textbf{x}_\perp) \rangle$, $N_{events}=$ %d' %n_events)
    plt.xlabel(r'$s = |x|$  [fm] (y = 0)')
    plt.ylabel(r'\textbf{Thickness} $T(s)$ [fm$^{-2}$]')
    plt.xlim([0,12.0])
    plt.legend(framealpha=0.2)
    plt.savefig('../../plots/thicknessMC.pdf', format='pdf', dpi=300)

