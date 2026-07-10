import numpy as np
from scipy.special import exp1 as E1
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import math
import sys
import os

sys.path.append(os.path.abspath("../"))

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": True,
    "lines.antialiased": True,
})

class params:
    def __init__(self, Q02, NC, alpha_S, BG, m, root_sNN, la, max_r, delt=1.0):
        self.Q02 = Q02
        self.CF = NC * NC - 1 / (2.0 * NC)
        self.alpha_S = alpha_S
        self.BG = 0.156
        self.s0 = 2.0 * np.pi * BG
        self.m = m
        self.sqsNN = root_sNN
        self.la = la
        self.delt = delt
        self.max_r = max_r

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
        print(f"lambda : {self.la}")
        print(f"delta : {self.delt}")
        print(f"max radius : {self.max_r} fm")


class atom:
    def __init__(self, A, R, a):
        self.A = A
        self.R = R
        self.a = a

#===================================================================================
# Thickness Function
def nuclei_sampling(params, atom):
    positions = [None] * atom.A
    i = 0
    while i < atom.A:
        x, y, z = np.random.uniform(-atom.R - params.max_r * atom.a, atom.R + params.max_r * atom.a, size=(3,))

        r = np.sqrt(x*x + y*y + z*z)

        #Woods - Saxon distribution from the slides
        rho = 1.0 / (1.0 + np.exp((r - atom.R) / atom.a))

        #Monte Carlo Sampling by rejection
        if np.random.rand() < rho:
            positions[i] = [x, y, z]
            i += 1

    return np.array(positions)

def thickness_A(X, Y, nucleons, params):
    # T_A = \sum_i^A T_N(x_\perp - x_i) Eq 9 PRC 109
    T = np.zeros_like(X)
    for x_i, y_i, _ in nucleons:
        T += (1.0 / params.s0) * np.exp( -( (X - x_i)**2 + (Y - y_i)**2 ) / (2.0 * params.BG) )
        T[T < 1.0e-12] = 0.0 #cut off of values that are virtually zero
    return T
#===================================================================================

def QF2(x, T, params):
    val = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if x[i][j] < 1.0 and x[i][j] > 1.0e-10:
                val[i][j] = params.Q02 * np.pow(x[i][j], -params.la) * np.pow(1 - x[i,j], params.delt) * T[i,j] * params.s0
            else:
                val[i][j] = 0.0
    return val

def low_x(eta_s, T, params):
    x = np.pow( params.Q02 * params.s0 * T * np.exp(2.0 * eta_s) / params.sqsNN ** 2, 1.0 / (2.0 + params.la))
    return x

def gluon_density(Q12, Q22, params, min_denom = 0.2, max_density=None):
    denom = Q12 + Q22
    denom = np.maximum(denom, min_denom) #Prevents large denominators that diverges

    val = np.zeros_like(Q12)

    for i in range(Q12.shape[0]):
        for j in range(Q12.shape[1]):
            m2Den = params.m**2 / denom[i][j]
            pref = (params.CF / (np.pi ** 2 * params.alpha_S)) * Q12[i][j] * Q22[i][j] / denom[i][j]**3
            sum1 = 0.5 * (Q12[i][j]**2 + Q22[i][j]**2)
            sum2 = - Q12[i][j] * Q22[i][j] * m2Den
            sum3 = ( Q12[i][j] * Q22[i][j] * (1 + 0.5 * m2Den**2) - 0.5 * ( Q12[i][j] - Q22[i][j ])**2 * m2Den ) * E1(m2Den) * np.exp(m2Den)

    val[val < 0] = 0.0

    val = np.minimum(val, max_density) if max_density is not None else val
    return val

def MCGB_step(eta_s, T1, T2, params):
    x1 = low_x(-eta_s, T1, params)
    x2 = low_x(eta_s, T2, params)

    Q12 = QF2(x1, T1, params)
    Q22 = QF2(x2, T2, params)

    ngt0 = gluon_density(Q12, Q22, params)

    N = np.random.poisson(lam=dA*ngt0)

    return N

if __name__=='__main__':
    pars = params(0.152, 3, 0.2, 0.156, 0.1, 2760, 0.215, 12.0)
    atm = atom(208, 6.5, 0.5) #Pb
    nucleons1 = nuclei_sampling(pars, atm)
    nucleons2 = nuclei_sampling(pars, atm)
    #=================================
    # Lattice
    #=================================
    dA = 0.1
    dx = np.sqrt(dA)
    dy = np.sqrt(dA)

    x = np.arange(-pars.max_r, pars.max_r + dx, dx)
    y = np.arange(-pars.max_r, pars.max_r + dy, dy)
    Nx = x.size
    Ny = y.size
    X, Y = np.meshgrid(x, y, indexing='ij') #lattice

    T1 = thickness_A(X,Y, nucleons1, pars)
    T2 = thickness_A(X,Y, nucleons2, pars)

    N = MCGB_step(1.0, T1, T2, pars)
    print(N.mean())
    pars.print()
    """
    A1 = np.sum(T1*dA)
    A2 = np.sum(T2*dA)
    print(f"Integration of the Thickness: T1 = {A1:.0f}")
    print(f"Integration of the Thickness: T2 = {A2:.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True, gridspec_kw={"wspace": 0.04})
    vmax = max(T1.max(), T2.max())
    vmin = min(T1.min(), T2.min())

    shared_levels = np.linspace(vmin, vmax, 15)
    im1 = ax1.contourf(X, Y, T1, levels=shared_levels, cmap='inferno')
    ax1.set_xlabel('x (fm)')
    ax1.set_ylabel('y (fm)')
    ax1.set_aspect('equal')

    im2 = ax2.contourf(X,Y, T2, levels=shared_levels, cmap='inferno')
    ax2.set_xlabel('x (fm)')
    ax2.set_ylabel('y (fm)')
    ax2.set_aspect('equal')

    cbar = fig.colorbar(im2, ax=ax2)
    cbar.set_label('Thickness $T$ (fm$^{-2}$)', fontsize=12, rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig("../../plots/x12.pdf", format='pdf')
    """
