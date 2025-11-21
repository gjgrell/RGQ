import numpy as np
from scipy.optimize import brentq

def xi_A_values(file):
    data = np.loadtxt(file, delimiter=',')
    log_xi = data[:,0]
    log_A = data[:,1]

    xi_values = 10**(log_xi)
    A_values = 10**(log_A)
    
    return xi_values, A_values

def xi_r0(L, r0, n0):
    return L / (n0 * r0**2)

def xi_rmax(L, beta, r, r0, n0):
    return L / (n0 * r0**beta * r**(2-beta))

def F(L,beta, r, r0, n0):
    xi = xi_rmax(L,beta, r, r0, n0)
    xi_values, A_values = xi_A_values("Si_he.csv")
    return np.interp(xi, xi_values, A_values)

def Fdiff(n0,L,r0,rmax,beta):
    return F(L,beta, r0, r0, n0) - F(L,beta, r0, rmax, n0)

def find_min_max_x(n0_scan,L,r0,rmax,beta):
    values = [Fdiff(n,L,r0,rmax,beta) for n in n0_scan]

    min_index = min(range(len(values)), key=lambda i: values[i])
    max_index = max(range(len(values)), key=lambda i: values[i])

    return min_index, max_index, values[min_index], values[max_index]


def F_n0_root(L, beta, r, r0, rmax):

    n0_scan = np.logspace(-5, 5, 1000)

    n0_lower = n0_scan[find_min_max_x(n0_scan,L,r0,rmax,beta)[0]]
    n0_upper = n0_scan[find_min_max_x(n0_scan,L,r0,rmax,beta)[1]]

    n0_solution = brentq(Fdiff,n0_lower,n0_upper,args=(L,r0,rmax,beta))
    
    return F(L,beta, r, r0, n0_solution)