import numpy as np
from scipy.special import exp1 as E1
import matplotlib.pyplot as plt

#=================================
# Global Parameters
#=================================
NC = 3
alpha_S = 0.3
CF = NC**2 - 1 / (2 * NC)
BG = 0.156 #fm^2
m = 0.3 #GeV
sigma_0 = 2.0 * np.pi * BG
root_sNN = 2796 # GeV
Q02 = 1.0 #GeV^2
la = 0.275
#=================================
# Pb Nuclei
#================================
A = 208
R = 6.5
a = 0.5

#=================================
# Lattice
#=================================
dA = 0.1
dx = np.sqrt(dA)
dy = np.sqrt(dA)

max_r = 12.0
x = np.arange(-max_r, max_r + dx, dx)
y = np.arange(-max_r, max_r + dy, dy)
Nx = x.size
Ny = y.size
X, Y = np.meshgrid(x, y, indexing='ij') #lattice

#===================================================================================
# Thickness Function
def thickness_N(X, Y):
    return (1.0 / (2.0 * np.pi * BG)) * np.exp(- (X**2 + Y**2) / (2.0 * BG))

def nuclei_sampling():
    positions = []
    while len(positions) < A:
        x = np.random.uniform(-R - max_r * a, R + max_r * a)
        y = np.random.uniform(-R - max_r * a, R + max_r * a)
        z = np.random.uniform(-R - max_r * a, R + max_r * a)

        r = np.sqrt(x*x + y*y + z*z)

        #Woods - Saxon distribution from the slides
        rho = 1.0 / (1.0 + np.exp((r - R) / a))

        #Monte Carlo Sampling by rejection
        if np.random.rand() < rho:
            positions.append([x, y, z])

    return np.array(positions)

def thickness_A(TN_func, nucleons, X, Y):
    T = np.zeros_like(X)
    for x_i, y_i, _ in nucleons:
        T += thickness_N(X - x_i, Y - y_i)
    return T
#===================================================================================

def QA2(x, Q02, la, T, s0, delt):
    return Q02 * np.pow(x, -la) * np.pow(1 - x, delt) * T * s0

def low_x(eta_s, Q02, la, T, s0, root_sNN):
    x = np.pow( Q02 * s0 * T * np.exp(2.0 * eta_s) / root_sNN ** 2, 1.0 / (2.0 + la))
    return np.where(x >= 1, x, 0.0)

def gluon_density(Q12, Q22):
    denom = Q12 + Q22 + 0.001
    assert denom.any() != 0, "A denominator equal to zero in gluon_density()"
    m2Den = m**2 / denom
    pref = (CF / (np.pi ** 2 * alpha_S)) * Q12 * Q22 / denom**3
    sum1 = 0.5 * (Q12**2 + Q22**2)
    sum2 = - Q12 * Q22 * m2Den
    sum3 = np.exp(m2Den) * E1(m2Den) * (Q12 * Q22 * (1 + 0.5 * m2Den**2)
                                        - 0.5 * (Q12 - Q22)**2 * m2Den )

    return pref * (sum1 + sum2 + sum3)


def MCGB(eta_s, X, Y, TA_func, TN_func, N_func):
    nucleons = N_func()
    T1 = TA_func(TN_func, nucleons , X, Y)
    x1 = low_x(-eta_s, Q02, la, T1, sigma_0, root_sNN)
    T2 = TA_func(TN_func, nucleons , X, Y)
    x2 = low_x(eta_s, Q02, la, T2, sigma_0, root_sNN)

    Q12 = QA2(x1, Q02, la, T1, sigma_0, 1.0)
    Q22 = QA2(x2, Q02, la, T2, sigma_0, 1.0)

    ngt0 = gluon_density(Q12, Q22)

    print(ngt0)
    N = np.random.poisson(dA*ngt0)

    return N

N = MCGB(0.0, X, Y, thickness_A, thickness_N, nuclei_sampling)

