import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.integrate import trapezoid, cumulative_trapezoid

HBARC = 0.19731          # GeV fm
K_G = 1.25
SIGMA_0 = 0.156 * 2 * np.pi   # fm^2 ; Ts = T(fm^-2) * SIGMA_0 (dimensionless)

def saturation_scale(x, Q02, la, delt, Ts0):
    #truncation up to machine precission
    xc = np.clip(x, 1e-15, 1.0)
    return np.where(x >= 1.0, 0.0, Q02 * Ts0 * (1.0 - xc)**delt / xc**la)


def single_incl_spec_gluons_cell_GBW(y, px, py, rsNN, Ts1, Ts2, m=0.1,
                                     delt=5.6, la=0.215,
                                     Q02=0.152, alpha_S=0.3,
                                     CF=4./3.):
    p2 = px**2 + py**2
    pT = np.sqrt(p2 + m**2) #no es un problema para low-x self-consistent
    x1 = pT * np.exp(y) / rsNN
    x2 = pT * np.exp(-y) / rsNN
    Q12 = saturation_scale(x1, Q02, la, delt, Ts1)
    Q22 = saturation_scale(x2, Q02, la, delt, Ts2)
    k = Q12 + Q22
    bounds = (x1 < 1.0) & (x2 < 1.0) & (k > 0.0)
    ks = np.where(bounds, k, 1.0)
    pref = CF * Q12 * Q22 / (2.0 * np.pi**3 * alpha_S * ks**3)
    A = 2.0 * Q12 * Q22
    B = (Q12 - Q22)**2 / ks
    C = Q12 * Q22 / ks**2
    val = pref * np.exp(-p2 / ks) * (A + B * p2 + C * p2**2) / (p2 + m**2)
    return np.where(bounds, val, 0.0)


def log_target_pt(y, px, py, rsNN, Ts1, Ts2, m=0.1, delt=5.6,
                  la=0.215, Q02=0.152, alpha_S=0.3,
                  CF=4./3.):
    p2 = px**2 + py**2
    pT = pt.sqrt(p2 + m**2)
    x1 = pT * pt.exp(y) / rsNN
    x2 = pT * pt.exp(-y) / rsNN
    inside = pt.and_(x1 < 1.0, x2 < 1.0)
    x1c = pt.minimum(x1, 1.0 - 1e-9)
    x2c = pt.minimum(x2, 1.0 - 1e-9)
    Q12 = Q02 * Ts1 * (1.0 - x1c)**delt / x1c**la
    Q22 = Q02 * Ts2 * (1.0 - x2c)**delt / x2c**la
    k = Q12 + Q22
    A = 2.0 * Q12 * Q22
    B = (Q12 - Q22)**2 / k
    C = Q12 * Q22 / k**2
    logpi = (np.log(CF / (2.0 * np.pi**3 * alpha_S))
             + pt.log(Q12) + pt.log(Q22) - 3.0 * pt.log(k) - p2 / k
             + pt.log(A + B * p2 + C * p2**2) - pt.log(p2 + m**2))
    return pt.switch(inside, logpi, -np.inf)


def build_model(rsNN, y_range=(-1.0, 1.0)):
    with pm.Model() as model:
        #place holders
        Ts1 = pm.Data("Ts1", np.array(1.0))
        Ts2 = pm.Data("Ts2", np.array(1.0))
        y = pm.Uniform("y", lower=y_range[0], upper=y_range[1])
        px = pm.Flat("px")
        py = pm.Flat("py")
        pm.Potential("log_target", log_target_pt(y, px, py, rsNN, Ts1, Ts2))
    return model


def MCMC_step(model, Ts1, Ts2, N_g, tune=500, chains=2, seed=None):
    """N_g (y, px, py) draws for a cell with thicknesses Ts1, Ts2."""
    with model:
        pm.set_data({"Ts1": Ts1, "Ts2": Ts2})
        idata = pm.sample(draws=int(np.ceil(N_g / chains)), tune=tune,
                          chains=chains, cores=1, random_seed=seed,
                          initvals={"y": 0.0, "px": 0.5, "py": 0.1},
                          progressbar=False, compute_convergence_checks=False)
    post = idata.posterior
    out = {k: np.asarray(post[k].values).reshape(-1)[:N_g]
           for k in ("y", "px", "py")}
    out["pt"] = np.hypot(out["px"], out["py"])
    return out, idata


def _marginal_grid(rsNN, Ts1, Ts2, y_range, n_y=61, n_p=800, p_lo=1e-3, p_hi=20.0):
    ys = np.linspace(*y_range, n_y)
    ps = np.geomspace(p_lo, p_hi, n_p)
    Y, P = np.meshgrid(ys, ps, indexing="ij")
    f = 2.0 * np.pi * P**2 * single_incl_spec_gluons_cell_GBW(Y, P, 0.0, rsNN, Ts1, Ts2)
    return ys, ps, f


def cell_multiplicity(rsNN, Ts1, Ts2, da, y_range=(-1.0, 1.0)):
    """Poisson mean  K_g * da * Int dy d^2p pi / hbarc^2   (da in fm^2)."""
    ys, ps, f = _marginal_grid(rsNN, Ts1, Ts2, y_range)
    dN_dy = trapezoid(f, np.log(ps), axis=1)          # GeV^2
    return K_G * trapezoid(dN_dy, ys) / HBARC**2 * da

if __name__ == "__main__":
    import time
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

    rsNN = 2760.0
    y_range = (-8.0, 8.0)
    T1 = T2 = 1.5 # fm^-2
    Ts1, Ts2 = T1 * SIGMA_0, T2 * SIGMA_0
    da = 0.1 * 0.1
    rng = np.random.default_rng(42)
    pts = rng.uniform([-1, -3, -3], [1, 3, 3], size=(5, 3))

    ys, ps, f = _marginal_grid(rsNN, Ts1, Ts2, y_range)
    f_p = trapezoid(f, ys, axis=0)
    norm = trapezoid(f_p, np.log(ps))
    cdf_p = cumulative_trapezoid(f_p, np.log(ps), initial=0.0) / norm
    mean_pt = trapezoid(f_p * ps, np.log(ps)) / norm
    f_y = trapezoid(f, np.log(ps), axis=1)
    mean_y = trapezoid(f_y * ys, ys) / trapezoid(f_y, ys)
    lam = cell_multiplicity(rsNN, Ts1, Ts2, da, y_range)
    print(f"numerical <pT> = {mean_pt:.4f} GeV, <y> = {mean_y:+.4f}, "
          f"cell Poisson mean = {lam:.4f}")

    model = build_model(rsNN, y_range)
    t = time.time()
    mom, idata = MCMC_step(model, Ts1, Ts2, N_g=6000, chains=2, seed=3)
    print(f"    sampled <pT> = {mom['pt'].mean():.4f} +- "
          f"{mom['pt'].std() / np.sqrt(len(mom['pt'])):.4f} GeV, "
          f"<y> = {mom['y'].mean():+.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(mom["pt"], bins=60, range=(0, 6), density=True, alpha=0.5,
               label="PyMC")
    ax[0].plot(ps, f_p / ps / norm, "k", lw=1.5, label="numerical")
    ax[0].set_xlim(0, 6)
    ax[0].set_xlabel(r"$p_T$ [GeV]")
    ax[0].set_ylabel(r"$dN/dp_T$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_ylim([1e-4,10])
    ax[1].hist(mom["y"], bins=40, density=True, alpha=0.5, label="PyMC")
    ax[1].plot(ys, f_y / trapezoid(f_y, ys), "k", lw=1.5, label="numerical")
    ax[1].set_xlabel("y")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("../../plots/gluon_pymc.pdf", format="pdf",  dpi=300)
