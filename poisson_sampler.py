import numpy as np
from scipy.special import exp1 as E1
import matplotlib.pyplot as plt

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
    positions = []
    while len(positions) < atom.A:
        x = np.random.uniform(-atom.R - params.max_r * atom.a, atom.R + params.max_r * atom.a)
        y = np.random.uniform(-atom.R - params.max_r * atom.a, atom.R + params.max_r * atom.a)
        z = np.random.uniform(-atom.R - params.max_r * atom.a, atom.R + params.max_r * atom.a)

        r = np.sqrt(x*x + y*y + z*z)

        #Woods - Saxon distribution from the slides
        rho = 1.0 / (1.0 + np.exp((r - atom.R) / atom.a))

        #Monte Carlo Sampling by rejection
        if np.random.rand() < rho:
            positions.append([x, y, z])

    return np.array(positions)

def thickness_A(X, Y, nucleons, params):
    # T_A = \sum_i^A T_N(x_\perp - x_i) Eq 9 PRC 109
    T = np.zeros_like(X)
    for x_i, y_i, _ in nucleons:
        T += (1.0 / params.s0) * np.exp( -( (X - x_i)**2 + (Y - y_i)**2 ) / (2.0 * params.BG) )
    return T
#===================================================================================

def QA2(x, T, params):
    return params.Q02 * np.pow(x, -params.la) * np.pow(1 - x, params.delt) * T * params.s0

def low_x(eta_s, T, params):
    x = np.pow( params.Q02 * params.s0 * T * np.exp(2.0 * eta_s) / params.root_sNN ** 2, 1.0 / (2.0 + params.la))
    return np.where(x >= 1, x, 0.0)

def gluon_density(Q12, Q22, params):
    denom = Q12 + Q22
    assert denom.any() != 0, "A denominator is equal to zero in gluon_density()"
    m2Den = params.m**2 / denom
    pref = (params.CF / (np.pi ** 2 * params.alpha_S)) * Q12 * Q22 / denom**3
    sum1 = 0.5 * (Q12**2 + Q22**2)
    sum2 = - Q12 * Q22 * m2Den
    sum3 = np.exp(m2Den) * E1(m2Den) * (Q12 * Q22 * (1 + 0.5 * m2Den**2)
                                        - 0.5 * (Q12 - Q22)**2 * m2Den )

    return pref * (sum1 + sum2 + sum3)


def MCGB(eta_s, T1, T2):
    x1 = low_x(-eta_s, T1, params)
    x2 = low_x(eta_s, T2, params)

    Q12 = QA2(x1, T1, params)
    Q22 = QA2(x2, T2, params)

    ngt0 = gluon_density(Q12, Q22, params)

    N = np.random.poisson(dA*ngt0)

    return N

if __name__=='__main__':
    pars = params(1.0, 3, 0.3, 0.156, 0.3, 2760, 0.275, 12.0)
    atm = atom(208, 6.5, 0.5) #Pb
    nucleons = nuclei_sampling(pars, atm)
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

    T1 = thickness_A(X,Y, nucleons, pars)
    T2 = thickness_A(X,Y, nucleons, pars)

    pars.print()
    print(T1)
