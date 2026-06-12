import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
from numpy.polynomial.legendre import leggauss
from functools import partial

#Constants
A_Si = 3.24e-5 #silicon fraction
A_Mg = 3.98e-5 #magnesium fraction
A_S = 	1.32e-5 #sulfur


def get_A_fraction(Z):
    fractions = {
        "Si": A_Si,
        "Mg": A_Mg,
        "S": A_S
    }
    return fractions.get(Z)

def xi_A_values(file):
    data = np.loadtxt(file, delimiter=',')
    log_xi = data[:,0]
    log_A = data[:,1]

    return log_xi, log_A


def xi_rmax(L, r0, r, n0, beta):
    return L / (n0 * r0**beta * r**(2-beta))


def F_root(Z, nele, L, r0, r, beta, xi_max):
    log_xi, log_A = xi_A_values("xi_calc/"+str(Z)+"_"+str(nele)+".csv")
    
    #Root-calculated initial number density (as a function of xi - XSTSR)
    n0_solution = L / (xi_max * r0**(2))
    xi = L / (n0_solution * r0**(beta) * r**(2-beta))
    f = interpolate.interp1d(log_xi, log_A, fill_value='extrapolate')
    Froot = f(xi)
    return Froot


def F_root_bimodal(Z, nele, mixing, L, r0, r, beta, xi_max):
    #Root-calculated initial number density (as a function of xi - XSTSR)
    n0_solution = L / (xi_max * r0**(2))
    
    #Define bimodal distribution for n0 based on mixing factor
    n0_low = (2 * n0_solution) / (1 + 1 / (1 - mixing)) 
    n0_high = n0_low / (1 - mixing)
    
    #Calculate xi for low, high modes
    xi_low = L / (n0_low * r0**(beta) * r**(2-beta))
    xi_high = L / (n0_high * r0**(beta) * r**(2-beta))    
    
    log_xi, log_A = xi_A_values("xi_calc/"+str(Z)+"_"+str(nele)+".csv")
    
    f = interpolate.interp1d(log_xi, log_A, fill_value='extrapolate')
    Froot_low = f(xi_low)
    Froot_high = f(xi_high)
    Froot_high_log = 10**(Froot_high)
    Froot_low_log = 10**(Froot_low)

    Froot = (Froot_low_log + Froot_high_log) / 2
    return Froot


def get_n(r, Z, nele, mixing, L, r0, beta, xi_max):
    
    #Root-calculated initial number density (as a function of xi - XSTSR)
    n0_solution = L / (xi_max * r0**(2))
    
    #Define bimodal distribution for n0 based on mixing factor
    n0_low = (2 * n0_solution) / (1 + 1 / (1 - mixing)) 
    n0_high = n0_low / (1 - mixing)
    
    #Calculate xi for low, high modes
    xi_low = L / (n0_low * r0**(beta) * r**(2-beta))
    xi_high = L / (n0_high * r0**(beta) * r**(2-beta))    
    
    log_xi, log_A = xi_A_values("xi_calc/"+str(Z)+"_"+str(nele)+".csv")
    
    f = interpolate.interp1d(log_xi, log_A, fill_value='extrapolate')
    Froot_low = f(xi_low)
    Froot_high = f(xi_high)
    Froot_high_log = 10**(Froot_high)
    Froot_low_log = 10**(Froot_low)

    Froot = (Froot_low_log + Froot_high_log) / 2
    
    #Column density approximation model
    n_low = 10**(Froot_low) * get_A_fraction(Z) * n0_low * (r0/r)**(beta) 
    n_high = 10**(Froot_high) * get_A_fraction(Z) * n0_high * (r0/r)**(beta) 
        
    #Take average for He-like ion density
    n = (n_low + n_high) / 2
    return n
    

def smax_cone(r0,theta0,phi0,i,alpha):
    mu = (np.sin(theta0)*np.cos(phi0)*np.cos(i)+np.cos(theta0)*np.sin(i))
    A = np.sin(i)**2 - np.cos(alpha)**2
    B = 2.0*r0*(np.cos(theta0)*np.sin(i)-np.cos(alpha)**2*mu)
    C = r0**2 * (np.cos(theta0)**2 - np.cos(alpha)**2)
    disc = B**2 - 4.0*A*C
    return (-B-np.sqrt(disc))/(2.0*A),mu

NQUAD=32
GL_X,GL_W = leggauss(NQUAD)
def obs_tau(r0,theta0,phi0,i,alpha,sigma,n_r):
    smax,mu = smax_cone(r0,theta0,phi0,i,alpha)
    s = 0.5*smax*(GL_X+1.0)
    r = np.sqrt(r0*r0+2.0*r0*mu*s + s**2)
    return 0.5*smax*np.sum(GL_W*n_r(r))*sigma
