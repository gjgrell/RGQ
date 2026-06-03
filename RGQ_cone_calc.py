import numpy as np
from scipy.integrate import simpson
from scipy import interpolate
from scipy.interpolate import interpn
from scipy import special
import sys
import pandas as pd
from scipy.optimize import minimize

from calc_Z_emit_ratio import *
from line_params import *
from frac_calc import *

#Constants
c = 3e10 #speed of light (cm/s)
m_e  = 510998.9 / (3e10)**2 #electron mass [eV / c^2]
m_p = 1.67e-24 #proton mass (grams)
#me = 9.11e-28 #grams
hbar = 6.582e-16 #Planck constant (eV*s)
h = 4.135e-15 #Planck constant (eV*s)
e2 = (hbar * c) / 137 #eV*cm
A_si = 3.24e-5 #silicon fraction


data_NIST = np.loadtxt('NIST_energies_qrst_wxyz.txt')
columns = ['Z', 'q', 'r', 's', 't', 'w','x','y','z']
index = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
df = pd.DataFrame(data_NIST,index=index, columns=columns)


#get the number for the element
def get_number(Z):
    element_num= {"C":6,"N":7,"O":8,"Ne":10,"Mg":12,"Si":14,"S":16,"Ar":18,"Ca":20,"Cr":24,"Mn":25,"Fe":26,"Ni":28}
    return element_num[Z]


# Power law function
def power_law(x, a, b):
    return a * np.power(x, -b)

#Voigt function    
def voigt(x, x0, sigma, gamma):
    return np.real(special.wofz((x - x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

#Calculates observed (global) R ratio averaged over entire medium factoring in plasma geometry and observer position (FULL APPROX)
#Assuming mixed cone for geometry
#Assumming cold photoionized plasma (ne << nc, phi_UV << phi_c)     
def R_analytic_global_cone(Z, v, mixing, gamma, theta_inner, theta_outer, inner_factor, R0, R, L, beta, ximax):
    
    #____
    #Relevant ratios F, B, K for analytic R ratio calculation
    ratios = get_ratios(Z, 0)
    F = ratios[0] #Ratio of collisional population of z / (x + y)
    B = ratios[1] #Effective branching ratio
    B_x = ratios[2] #Branching ratio for x to multiply by escape probability
    B_y = ratios[3] #Branching ratio for y to multiply by escape probability
    UV_x = ratios[4] #UV photoexcitation rate * branching ratio for x
    UV_y = ratios[5]  #UV photoexcitation rate * branching ratio for y
    K = ratios[6] #Ratio of radiative recombination rates of w / (x + y)
    R_0 = ratios[7] #FAC-calculated z / (x + y) in absence of 2s-2p UV photoexcitation rate
    
    #____
    #Line profiles
    #He-like x emitting
    He_x_params = get_params_He(Z, 'x')
    x_centroid_energy = df.loc[Z,'x']
    x_lorentz_gamma = He_x_params[2] * hbar #eV

    #He-like y emitting
    He_y_params = get_params_He(Z, 'y')
    y_centroid_energy = df.loc[Z, 'y']
    fac_y_osc_strength = He_y_params[1] #oscillator strength
    y_lorentz_gamma = He_y_params[2] * hbar #eV

    #Li-like s absorbing
    Li_s_params = get_params_Li(Z, 's')
    s_centroid_energy = df.loc[Z,'s']
    fac_s_osc_strength = Li_s_params[1] #oscillator strength
    s_lorentz_gamma = Li_s_params[2] * hbar #eV

    #Li-like t absorbing
    Li_t_params = get_params_Li(Z, 't')
    t_centroid_energy = df.loc[Z,'t']
    fac_t_osc_strength = Li_t_params[1] #oscillator strength
    t_lorentz_gamma = Li_t_params[2] * hbar #eV

    #Energy distribution
    e_lower = (y_centroid_energy - 5)
    e_upper = (y_centroid_energy + 5)
    e_grid = np.linspace(e_lower, e_upper, 100) 

    #Line profiles
    sigma_em_x = (v * 1e5 / c) * x_centroid_energy
    sigma_em_y = (v * 1e5 / c) * y_centroid_energy

    sigma_em_s = (v * 1e5 / c) * s_centroid_energy
    sigma_em_t = (v * 1e5 / c) * t_centroid_energy

    phi_x = voigt(e_grid, x_centroid_energy, sigma_em_x, x_lorentz_gamma)
    phi_y = voigt(e_grid, y_centroid_energy, sigma_em_y, y_lorentz_gamma)
    phi_s = voigt(e_grid, s_centroid_energy, sigma_em_s, s_lorentz_gamma)
    phi_t = voigt(e_grid, t_centroid_energy, sigma_em_t, t_lorentz_gamma)

    #omega - Radiative decay probabilities 
    fac_ai_rate_s = get_ai_rates(Z, 's')
    fac_omega_s = Li_s_params[2] / (Li_s_params[2] + fac_ai_rate_s)

    fac_ai_rate_t = get_ai_rates(Z, 't')
    fac_omega_t = Li_t_params[2] / (Li_t_params[2] + fac_ai_rate_t)

    #Absorption cross section for s
    sigma_s = h * (np.pi * e2 / (m_e * c)) * (fac_s_osc_strength * phi_s) * (1 - fac_omega_s)
    sigma_t = h * (np.pi * e2 / (m_e * c)) * (fac_t_osc_strength * phi_t) * (1 - fac_omega_t)
    sigma_abs = sigma_s + sigma_t #energy dimensions
    
    
    #___
    #Conical geometry (spherical coordinates)
    #Inner and maximum radii
    R0 = R0 * 3.086e+18 #cm-3
    R = R * 3.086e+18 #cm-3
    
    #Conical geometry (spherical coordinates for volume integration)
    theta_lin = np.linspace(0, theta_outer, 10) #opening angle 
    phi_lin = np.linspace(0, 2*np.pi, 10) #azimuthal angle 
    r_lin = np.linspace(R0, R, 20) #radial distribution along LoS
    
    theta_grid, phi_grid, r_grid = np.meshgrid(theta_lin, phi_lin, r_lin, indexing='ij') #form 3D meshgrid for integration

    endpoint = (np.sqrt(np.cos(theta_grid)**2 * np.tan(theta_outer)**2) - np.sin(theta_grid)**2 * np.sin(phi_grid)**2)
    yz = np.sqrt(np.sin(theta_grid)**2 * np.sin(phi_grid)**2 + np.cos(theta_grid)**2)
        
    #input parameter for bicone inner radius - theta_inner
    #verify theta_inner should be < alpha (if statement)
    if (theta_inner > theta_outer):
        sys.exit("ERROR: Inner angle should be less than outer angle")
    
    n_factor = np.ones_like(theta_grid)
    n_factor[theta_grid <= theta_inner] = inner_factor
    
    #___
    #He-like, Li-like density derivations 
    #Froot - ion fraction of He,Li based on root-calculated initial density n0 (as a function of xi - XSTAR)
    #He-like
    
    #define n = n0_solution * n_factor * (R0/r_grid)**(beta)
    #n_he = 10**(Froot_he) * A_si * n
    
    log_xi_he, log_A_he = xi_A_values("Si_he.csv")
    n0_solution = L / (ximax * R0**(2))
    n_theta = n0_solution * n_factor * (R0/r_grid)**(beta)
    
    xi = L / (n_theta * (r_grid)**2)
    f_he = interpolate.interp1d(log_xi_he, log_A_he, fill_value='extrapolate')
    Froot_he = f_he(xi)
    
    n_he = 10**(Froot_he) * A_si * n_theta
    N_he = simpson(n_he, r_lin, axis = 2)[:,0]
    #print("NHe = " +f"{N_he:.2E}"+ " cm-2")
    
    log_xi_li, log_A_li = xi_A_values("Si_li.csv")    
    f_li = interpolate.interp1d(log_xi_li, log_A_li, fill_value='extrapolate')
    Froot_li = f_li(xi)
    
    n_li = 10**(Froot_li) * A_si * n_theta
    N_li = simpson(n_li, r_lin, axis = 2)[:,0]
    #print("NLi = " +f"{N_li:.2E}"+ " cm-2")
    
    #____
    #tot_beta_Li, tot_beta_He

    #phi = photoionization cross sections * ionizing radiation
    egrid_Li = []
    pi_sig_Li = []
    data_pi_Li = np.loadtxt('facfiles/RR_cs_Li/'+Z+'03_rr_cs_Li_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_Li)):
        egrid_Li.append(data_pi_Li[i][0])
        pi_sig_Li.append(data_pi_Li[i][2])

    y_Li = power_law(egrid_Li, 1, gamma)
    #beta_Li = y_Li * (1 - np.exp(-1*np.array(pi_sig_Li) * 1e-20 * N_li))
    tot_beta_Li_theta = np.zeros_like(theta_lin)
    
    for i in range(len(theta_lin)):
        beta_Li_theta = y_Li * (1 - np.exp(-1*np.array(pi_sig_Li) * 1e-20 * N_li[i]))
        tot_beta_Li_theta[i] = simpson(beta_Li_theta, egrid_Li)
        
    tot_beta_Li = simpson(tot_beta_Li_theta, theta_lin) #FAC output units: 1e20 cm-2
    
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0])
        pi_sig_He.append(data_pi_He[i][2])

    y_He = power_law(egrid_He, 1, gamma)
    #beta_He = y_He * (1 - np.exp(-1*np.array(pi_sig_He) * 1e-20 * N_he))
    tot_beta_He_theta = np.zeros_like(theta_lin)
    
    sigma_y = h * (np.pi * e2 / (m_e * c)) * fac_y_osc_strength * phi_y
    I_y = (y_centroid_energy)**(-1 * gamma)
    tau_y = np.zeros_like(theta_lin)
    
    for i in range(len(theta_lin)):
        beta_He_theta = y_He * (1 - np.exp(-1*np.array(pi_sig_He) * 1e-20 * N_he[i]))
        tot_beta_He_theta[i] = simpson(beta_He_theta, egrid_He)
        
        #y optical depth
        tau_y = sigma_y * N_he[i]
    
        #y photoexcitation rate
        dPhi_y = (1 - np.exp(-1*tau_y))
        Phi_y = I_y * simpson(dPhi_y, x=e_grid)
    
    tot_beta_He = simpson(tot_beta_He_theta, theta_lin) #FAC output units: 1e20 cm-2
    
    #____
    #A
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))

    #____
    #Fraction of radiative recombinations populating level i in recombined ion
    x_i = A / K

    #S - Ionization rate parameter
    S = tot_beta_Li / (x_i * tot_beta_He)
    
        
    #_____
    #y cross section for photoexcitation rate calculation
    sigma_y = h * (np.pi * e2 / (m_e * c)) * fac_y_osc_strength * phi_y    
    I_y = (y_centroid_energy)**(-1 * gamma)
    
    #Calculate RAD-corrected total photoionization rate
    #total photoionization rate = 4 * pi * Omega * r0^2 * integral((J/E) * (sigma / sigma_tot)) * (1 - e^(-tau)) dE)
    P_rad_x = np.zeros_like(r_grid)
    P_rad_y = np.zeros_like(r_grid)
    Utilde = np.zeros_like(r_grid)

    for i in range(len(theta_lin)):
        for j in range(len(phi_lin)):
            for k in range(len(r_lin)):
                tau = n_li[i,j,k] * sigma_abs * (r_grid[i,j,k] * endpoint[i,j,k])
                tau_y = n_he[i,j,k] * sigma_y * (r_grid[i,j,k] * endpoint[i,j,k])
                
                P_rad_y[i,j,k] = simpson(phi_y * (1 - np.exp(-1 * tau)), e_grid, axis = 0)
                P_rad_x[i,j,k] = simpson(phi_x * (1 - np.exp(-1 * tau)), e_grid, axis = 0)
                
                #y Photoexcitation rate
                dPhi_y = (1 - np.exp(-1 * tau_y))
                Phi_y = I_y * simpson(dPhi_y, x=e_grid)
                
                #RAD-modified U
                Utilde[i,j,k] = (B_y * Phi_y / (x_i * tot_beta_He)) * (1 - P_rad_y[i,j,k])
            
    
    #Total photoionization rate (integrate over volume)
    bar_beta_He = simpson(simpson(simpson((1 / (r_grid/R0)**(beta)) * np.sin(theta_grid) * 10**(Froot_he), theta_lin, axis = 0), phi_lin, axis = 0), r_lin, axis = 0)
    Btilde_y = B_y * (1 - P_rad_y)
    Btilde_x = B_x * (1 - P_rad_x) 
    Btilde = (1./3.) * Btilde_y + (5./9.) * Btilde_x
    
    #RAD-modified total photoionization rate (integrated)
    tilde_beta_He = tot_beta_He * simpson(simpson(simpson(10**(Froot_he) * (R0/r_grid)**(beta) * np.sin(theta_grid) * (Btilde / B) * (1 + Utilde), theta_lin, axis=0),phi_lin,axis=0),r_lin,axis=0)

    Bz = 1 - B
    R0_prime = (F + S + Bz) / B

    R_ratio = R0_prime * tot_beta_He * bar_beta_He / tilde_beta_He  
    return float(R_ratio) 


#Calculates observed (global) Q ratio averaged over entire medium factoring in plasma geometry and observer position (FULL APPROX)
#Assumming cold photoionized plasma (ne << nc, phi_UV << phi_c) 
def Q_analytic_global_cone(Z, v, mixing, gamma, theta_inner, theta_outer, inner_factor, R0, R, L, beta, ximax):

    #Conical geometry (spherical coordinates)
    #Inner and maximum radii
    R0 = R0 * 3.086e+18 #cm-3
    R = R * 3.086e+18 #cm-3
    
    #Conical geometry (spherical coordinates for volume integration)
    theta_lin = np.linspace(0, theta_outer, 10) #opening angle 
    phi_lin = np.linspace(0, 2*np.pi, 10) #azimuthal angle 
    r_lin = np.linspace(R0, R, 20) #radial distribution along LoS
    
    theta_grid, phi_grid, r_grid = np.meshgrid(theta_lin, phi_lin, r_lin) #form 3D meshgrid for integration

    endpoint = (np.sqrt(np.cos(theta_grid)**2 * np.tan(theta_outer)**2) - np.sin(theta_grid)**2 * np.sin(phi_grid)**2)
    yz = np.sqrt(np.sin(theta_grid)**2 * np.sin(phi_grid)**2 + np.cos(theta_grid)**2)
    
    #input parameter for bicone inner radius - theta_inner
    #verify theta_inner should be < alpha (if statement)
    if (theta_inner > theta_outer):
        sys.exit("ERROR: Inner angle should be less than outer angle")
    
    n_factor = np.ones_like(theta_grid)
    n_factor[theta_grid <= theta_inner] = inner_factor
    
    #___
    #He-like, Li-like density derivations 
    #Froot - ion fraction of He,Li based on root-calculated initial density n0 (as a function of xi - XSTAR)
    #He-like
    
    log_xi_he, log_A_he = xi_A_values("Si_he.csv")
    n0_solution = L / (ximax * R0**(2))
    n_theta = n0_solution * n_factor * (R0/r_grid)**(beta)
    
    xi = L / (n_theta * (r_grid)**2)
    f_he = interpolate.interp1d(log_xi_he, log_A_he, fill_value='extrapolate')
    Froot_he = f_he(xi)
    
    n_he = 10**(Froot_he) * A_si * n_theta
    N_he = simpson(n_he, r_lin, axis = 2)[:,0]
    #print("NHe = " +f"{N_he:.2E}"+ " cm-2")
    
    log_xi_li, log_A_li = xi_A_values("Si_li.csv")    
    f_li = interpolate.interp1d(log_xi_li, log_A_li, fill_value='extrapolate')
    Froot_li = f_li(xi)
    
    n_li = 10**(Froot_li) * A_si * n_theta
    N_li = simpson(n_li, r_lin, axis = 2)[:,0]
    #print("NLi = " +f"{N_li:.2E}"+ " cm-2")
    
    #____
    #tot_beta_He - cone-averaged total photoionization rate [s-1]
    #beta = photoionization cross sections * ionizing radiation [s -1]
    #egrid - energy range starting from K-shell PI threshold energy [eV]
    
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0]) #energy range starting from K-shell edge
        pi_sig_He.append(data_pi_He[i][2]) #absorption cross section

    y_He = power_law(egrid_He, 1, gamma) #energy power-law function [spectral index gamma]
    tot_beta_He_theta = np.zeros_like(theta_lin) #array to store photoionization rates along LoS (theta)
        
    for i in range(len(theta_lin)):
        beta_He_theta = y_He * (1 - np.exp(-1*np.array(pi_sig_He) * 1e-20 * N_he[i])) #local photoionization rate (theta)
        tot_beta_He_theta[i] = simpson(beta_He_theta, egrid_He) #integrated over energy grid
    
    tot_beta_He = simpson(tot_beta_He_theta, theta_lin) #FAC output units: 1e20 cm-2
    
    #_____
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))
            
    #_____
    
    #Pi
    NLI_space,NHE_space,vel_space,qrw_ratio = get_q_r_w_ratio(get_number(Z))
    
    ##### FIX ######
    Pi = interpn((NLI_space,NHE_space,vel_space),qrw_ratio,[N_li,N_he,v])[0]
	
    #____
    #Q, S

    #He-like w line
    He_w_params = get_params_He(Z, 'w')
    w_centroid_energy = df.loc[Z,'w']
    fac_w_osc_strength = He_w_params[1] 
    w_lorentz_gamma = He_w_params[2] * hbar #eV

    #Energy distribution
    e_w_lower = (w_centroid_energy - 300)
    e_w_upper = (w_centroid_energy + 300)
    e_grid_w = np.linspace(e_w_lower, e_w_upper, 100000) 

    #w Line profile
    sigma_em_w = (v * 1e5 / c) * w_centroid_energy
    voigt_w = voigt(e_grid_w, w_centroid_energy, sigma_em_w, w_lorentz_gamma)
    sigma_w = h * (np.pi * e2 / (m_e * c)) * fac_w_osc_strength * voigt_w
    #tau_w = sigma_w * N_he

	#J_Ew / Ew
    I_w = (w_centroid_energy / E_0)**(-1 * gamma)
    
    for i in range(len(theta_lin)):
        for j in range(len(phi_lin)):
            for k in range(len(r_lin)):
                tau_w = n_he[i,j,k] * sigma_w * (r_grid[i,j,k] * endpoint[i,j,k])
                
                #w Photoexcitation rate
                dPhi_w = (1 - np.exp(-1 * tau_w))
                Phi_w = I_w * simpson(dPhi_y, x=e_grid_w)
    
    Upsilon = A * tot_beta_He / (Phi_w)
    Q = Pi / (1 + Upsilon)
        
    return float(Q)

    
#Calculates observed (global) G ratio averaged over entire medium factoring in plasma geometry and observer position (FULL APPROX)
#Assumming cold photoionized plasma (ne << nc, phi_UV << phi_c) 
def G_analytic_global_cone(Z, v, mixing, gamma, theta_inner, theta_outer, inner_factor, R0, R, L, beta, ximax):
    
    #____
    #Relevant ratios F, B, K for analytic R ratio calculation
    ratios = get_ratios(Z, 0)
    F = ratios[0] #Ratio of collisional population of z / (x + y)
    B = ratios[1] #Effective branching ratio
    B_x = ratios[2] #Branching ratio for x to multiply by escape probability
    B_y = ratios[3] #Branching ratio for y to multiply by escape probability
    UV_x = ratios[4] #UV photoexcitation rate * branching ratio for x
    UV_y = ratios[5]  #UV photoexcitation rate * branching ratio for y
    K = ratios[6] #Ratio of radiative recombination rates of w / (x + y)
    R_0 = ratios[7] #FAC-calculated z / (x + y) in absence of 2s-2p UV photoexcitation rate
    
  #____
    #FAC-calculated line parameters
    
    #He-like x emitting
    He_x_params = get_params_He(Z, 'x') #returns 0: line energy, 1: oscillator strength, 2: A rate
    x_centroid_energy = df.loc[Z,'x'] #reads from NIST line energy data file
    x_lorentz_gamma = He_x_params[2] * hbar #eV #gamma for Lorentzian component = hbar [eV*s] * A [s-1]
    
    #He-like y emitting
    He_y_params = get_params_He(Z, 'y')
    y_centroid_energy = df.loc[Z, 'y']
    fac_y_osc_strength = He_y_params[1] #oscillator strength
    y_lorentz_gamma = He_y_params[2] * hbar #eV
    
    #Li-like s absorbing
    Li_s_params = get_params_Li(Z, 's')
    s_centroid_energy = df.loc[Z,'s']
    fac_s_osc_strength = Li_s_params[1] #FAC-calculated oscillator strength
    s_lorentz_gamma = Li_s_params[2] * hbar #eV
    
    #Li-like t absorbing
    Li_t_params = get_params_Li(Z, 't')
    t_centroid_energy = df.loc[Z,'t']
    fac_t_osc_strength = Li_t_params[1] #oscillator strength
    t_lorentz_gamma = Li_t_params[2] * hbar #eV

    #Energy distribution
    e_lower = (y_centroid_energy - 300)
    e_upper = (y_centroid_energy + 300)
    e_grid = np.linspace(e_lower, e_upper, 100) 

    #Line profiles
    sigma_em_x = (v * 1e5 / c) * x_centroid_energy #sigma for Gaussian component
    sigma_em_y = (v * 1e5 / c) * y_centroid_energy
    
    sigma_em_s = (v * 1e5 / c) * s_centroid_energy
    sigma_em_t = (v * 1e5 / c) * t_centroid_energy
    
	#Voigt line profiles for He-like x,y, Li-like s, t
    phi_x = voigt(e_grid, x_centroid_energy, sigma_em_x, x_lorentz_gamma)
    phi_y = voigt(e_grid, y_centroid_energy, sigma_em_y, y_lorentz_gamma)
    phi_s = voigt(e_grid, s_centroid_energy, sigma_em_s, s_lorentz_gamma)
    phi_t = voigt(e_grid, t_centroid_energy, sigma_em_t, t_lorentz_gamma)
    
    #omega - Radiative decay probabilities 
    fac_ai_rate_s = get_ai_rates(Z, 's') #autoionization rate 
    fac_omega_s = Li_s_params[2] / (Li_s_params[2] + fac_ai_rate_s) #fluorescence yield
    
    fac_ai_rate_t = get_ai_rates(Z, 't')
    fac_omega_t = Li_t_params[2] / (Li_t_params[2] + fac_ai_rate_t)
    
    #___
    #Conical geometry (spherical coordinates)
    #Inner and maximum radii
    R0 = R0 * 3.086e+18 #cm-3
    R = R * 3.086e+18 #cm-3
    
    #Conical geometry (spherical coordinates for volume integration)
    theta_lin = np.linspace(0, theta_outer, 10) #opening angle 
    phi_lin = np.linspace(0, 2*np.pi, 10) #azimuthal angle 
    r_lin = np.linspace(R0, R, 20) #radial distribution along LoS
    
    theta_grid, phi_grid, r_grid = np.meshgrid(theta_lin, phi_lin, r_lin) #form 3D meshgrid for integration

    endpoint = (np.sqrt(np.cos(theta_grid)**2 * np.tan(theta_outer)**2) - np.sin(theta_grid)**2 * np.sin(phi_grid)**2)
    yz = np.sqrt(np.sin(theta_grid)**2 * np.sin(phi_grid)**2 + np.cos(theta_grid)**2)
    
    #input parameter for bicone inner radius - theta_inner
    #verify theta_inner should be < alpha (if statement)
    if (theta_inner > theta_outer):
        sys.exit("ERROR: Inner angle should be less than outer angle")
    
    n_factor = np.ones_like(theta_grid)
    n_factor[theta_grid <= theta_inner] = inner_factor
    
    #___
    #He-like, Li-like density derivations 
    #Froot - ion fraction of He,Li based on root-calculated initial density n0 (as a function of xi - XSTAR)
    #He-like
    
    log_xi_he, log_A_he = xi_A_values("Si_he.csv")
    n0_solution = L / (ximax * R0**(2))
    n_theta = n0_solution * n_factor * (R0/r_grid)**(beta)
    
    xi = L / (n_theta * (r_grid)**2)
    f_he = interpolate.interp1d(log_xi_he, log_A_he, fill_value='extrapolate')
    Froot_he = f_he(xi)
    
    n_he = 10**(Froot_he) * A_si * n_theta
    N_he = simpson(n_he, r_lin, axis = 2)[:,0]
    #print("NHe = " +str(N_he)+ " cm-2")
    
    log_xi_li, log_A_li = xi_A_values("Si_li.csv")    
    f_li = interpolate.interp1d(log_xi_li, log_A_li, fill_value='extrapolate')
    Froot_li = f_li(xi)
    
    n_li = 10**(Froot_li) * A_si * n_theta
    N_li = simpson(n_li, r_lin, axis = 2)[:,0]
    #print("NLi = " +str(N_li)+ " cm-2")
    
    #____
    #tot_beta_He, tot_beta_Li - cone-averaged total photoionization rate [s-1]
    #beta = photoionization cross sections * ionizing radiation [s -1]
    #egrid - energy range starting from K-shell PI threshold energy [eV]
    
    egrid_Li = [] 
    pi_sig_Li = []
    data_pi_Li = np.loadtxt('facfiles/RR_cs_Li/'+Z+'03_rr_cs_Li_K_edge.txt',skiprows=1,usecols=(0,3,4))
    
    for i in range(len(data_pi_Li)):
        egrid_Li.append(data_pi_Li[i][0]) #energy range starting from K-shell edge
        pi_sig_Li.append(data_pi_Li[i][2]) #absorption cross section

    y_Li = power_law(egrid_Li, 1, gamma) #energy power-law function [spectral index gamma]
    tot_beta_Li_theta = np.zeros_like(theta_lin) #array to store photoionization rates along LoS (theta)
    
    for i in range(len(theta_lin)):
        beta_Li_theta = y_Li * (1 - np.exp(-1*np.array(pi_sig_Li) * 1e-20 * N_li[i])) #local photoionization rate (theta)
        tot_beta_Li_theta[i] = simpson(beta_Li_theta, egrid_Li) #integrated over energy grid
        
    tot_beta_Li = simpson(tot_beta_Li_theta, theta_lin) #FAC output units: 1e20 cm-2    
            
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0]) #energy range starting from K-shell edge
        pi_sig_He.append(data_pi_He[i][2]) #absorption cross section

    y_He = power_law(egrid_He, 1, gamma) #energy power-law function [spectral index gamma]
    tot_beta_He_theta = np.zeros_like(theta_lin) #array to store photoionization rates along LoS (theta)
        
    for i in range(len(theta_lin)):
        beta_He_theta = y_He * (1 - np.exp(-1*np.array(pi_sig_He) * 1e-20 * N_he[i])) #local photoionization rate (theta)
        tot_beta_He_theta[i] = simpson(beta_He_theta, egrid_He) #integrated over energy grid
    
    tot_beta_He = simpson(tot_beta_He_theta, theta_lin) #FAC output units: 1e20 cm-2
    
    #___
    #Absorption cross section for s,t
    sigma_s = h * (np.pi * e2 / (m_e * c)) * (fac_s_osc_strength * phi_s) * (1 - fac_omega_s)
    sigma_t = h * (np.pi * e2 / (m_e * c)) * (fac_t_osc_strength * phi_t) * (1 - fac_omega_t)
    sigma_abs = sigma_s + sigma_t #s and t summed cross section, weighted by autoionization yield (1-omega)
    
    #____
    #A
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))
            
    #____
    #S
    
    #He-like w line
    He_w_params = get_params_He(Z, 'w')
    w_centroid_energy = df.loc[Z,'w']
    fac_w_osc_strength = He_w_params[1] 
    w_lorentz_gamma = He_w_params[2] * hbar #eV

    #Energy distribution
    e_w_lower = (w_centroid_energy - 5)
    e_w_upper = (w_centroid_energy + 5)
    e_grid_w = np.linspace(e_w_lower, e_w_upper, 100) 

    #w Line profile
    sigma_em_w = (v * 1e5 / c) * w_centroid_energy
    voigt_w = voigt(e_grid_w, w_centroid_energy, sigma_em_w, w_lorentz_gamma)
    sigma_w = h * (np.pi * e2 / (m_e * c)) * fac_w_osc_strength * voigt_w
    #tau_w = sigma_w * N_he

	#J_Ew / Ew
    I_w = (w_centroid_energy)**(-1 * gamma)
    
    for i in range(len(theta_lin)):
        for j in range(len(phi_lin)):
            for k in range(len(r_lin)):
                tau_w = n_he[i,j,k] * sigma_w * (r_grid[i,j,k] * endpoint[i,j,k])
                
                #w Photoexcitation rate
                dPhi_w = (1 - np.exp(-1 * tau_w))
                Phi_w = I_w * simpson(dPhi_w, x=e_grid_w)
    
    #x_i - Fraction of radiative recombinations populating level i in recombined ion
    x_i = A / K
    
    #S - Ratio of contribution to z from PI / contribution to x,y from RR
    S = tot_beta_Li / (x_i * tot_beta_He)
    Upsilon = A * tot_beta_He / (Phi_w)
    
    #_____
    #y Line profile (for y photoexcitation)
    sigma_y = h * (np.pi * e2 / (m_e * c)) * fac_y_osc_strength * phi_y    
    
    #tau_y = sigma_y * N_he    
    I_y = (y_centroid_energy)**(-1 * gamma)
    
    #Calculate RAD-corrected total photoionization rate
    #total photoionization rate = 4 * pi * Omega * r0^2 * integral((J/E) * (sigma / sigma_tot)) * (1 - e^(-tau)) dE)
    P_rad_x = np.zeros_like(r_grid)
    P_rad_y = np.zeros_like(r_grid)
    Utilde = np.zeros_like(r_grid)

    for i in range(len(theta_lin)):
        for j in range(len(phi_lin)):
            for k in range(len(r_lin)):
                tau = n_li[i,j,k] * sigma_abs * (r_grid[i,j,k] * endpoint[i,j,k])
                tau_y = n_he[i,j,k] * sigma_y * (r_grid[i,j,k] * endpoint[i,j,k])
                
                P_rad_y[i,j,k] = simpson(phi_y * (1 - np.exp(-1 * tau)), e_grid, axis = 0)
                P_rad_x[i,j,k] = simpson(phi_x * (1 - np.exp(-1 * tau)), e_grid, axis = 0)
                
                #y Photoexcitation rate
                dPhi_y = (1 - np.exp(-1 * tau_y))
                Phi_y = I_y * simpson(dPhi_y, x=e_grid)
                
                #RAD-modified U
                Utilde[i,j,k] = (B_y * Phi_y / (x_i * tot_beta_He)) * (1 - P_rad_y[i,j,k])
    
    #Total photoionization rate (integrate over volume)
    bar_beta_He = simpson(simpson(simpson((1 / (r_grid/R0)**(beta)) * np.sin(theta_grid) * 10**(Froot_he), theta_lin, axis = 0), phi_lin, axis = 0), r_lin, axis = 0)
    Btilde_y = B_y * (1 - P_rad_y)
    Btilde_x = B_x * (1 - P_rad_x) 
    Btilde = (1./3.) * Btilde_y + (5./9.) * Btilde_x
    
    #RAD-modified total photoionization rate (integrated)
    tilde_beta_He = tot_beta_He * simpson(simpson(simpson(10**(Froot_he) * (R0/r_grid)**(beta) * np.sin(theta_grid) * (Btilde / B) * (1 + Utilde), theta_lin, axis=0),phi_lin,axis=0),r_lin,axis=0)

    tilde_upsilon = A * bar_beta_He / (Phi_w)
    #G = (1 + F - B + S) / (K + (Phi_w * K / (tot_beta_He * A))) + (Btilde) / (A_0 * (K + (Phi_w * K / (tot_beta_He * A))))
    G = (1 / K) * (Upsilon / (Upsilon + 1)) * (1 + F + S - B * (1 - tilde_beta_He / (tot_beta_He * bar_beta_He)))  

    return float(G)


def get_RGQ_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax):
    R_ratio = R_analytic_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax)
    print(R_ratio)
    G_ratio = G_analytic_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax)
    print(G_ratio)
    Q_ratio = Q_analytic_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax)
    print(Q_ratio)
    return np.array([R_ratio,G_ratio,Q_ratio])
    
def chi_sq_RGQ_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax, RGQ_meas, RGQ_meas_err):
    RGQ_model = get_RGQ_global(Z, v, mixing, gamma, alpha, R0, R, L, beta, ximax)
    RGQ_meas = np.array(RGQ_meas)
    RGQ_meas_err = np.array(RGQ_meas_err)
    chi_arr = ((RGQ_meas-RGQ_model)**2)/RGQ_meas_err**2
    return np.sum(chi_arr)
    
#def chi_for_min_global(x, Z, mixing, gamma, alpha, L, RGQ_meas, RGQ_meas_err):
def chi_for_min_global(x, Z, gamma, alpha, L, RGQ_meas, RGQ_meas_err):
    print(x)
    vel = x[0]
    mixing = x[1]
    R0 = x[2]
    R = R0 * 100
    beta = x[4]
    ximax = x[5]
    RGQ_meas = np.array(RGQ_meas)
    RGQ_meas_err = np.array(RGQ_meas_err)
    chi = chi_sq_RGQ_global(Z, vel, mixing, gamma, alpha, R0, R, L, beta, ximax, RGQ_meas, RGQ_meas_err)
    print(chi)
    return chi

def solve_RGQ_global(Z, gamma, alpha, L, RGQ_meas, RGQ_meas_err):
    res = minimize(chi_for_min_global,x0=[250, 0.5, 0.4, 1.5, 10.5],args=(Z, gamma, alpha, L, RGQ_meas,RGQ_meas_err),bounds=[(100.1,399.9), (0.01, 0.99), (0.31, 0.51), (1.01, 1.99),(9.01, 11.01)])
    return res

def get_RGQ(Z, N_li, N_he, v, phi, nele, mixing, gamma):
    R = R_analytic_local(Z, N_li, N_he, v, phi, nele, mixing, gamma)
    # print(R)
    G = G_analytic_local(Z, N_li, N_he, v, phi, nele, mixing, gamma)
    # print(G)
    Q = Q_analytic_local(Z, N_li, N_he, v, phi, nele, gamma)
    # print(Q)
    return np.array([R,G,Q])


def chi_sq_RGQ(NHe,NLi,vel,mixing,gamma,Z,RGQ_meas,RGQ_meas_err):
    RGQ_model = get_RGQ(Z,NLi,NHe,vel,0,1,mixing,gamma)
    chi_arr = ((RGQ_meas-RGQ_model)**2)/RGQ_meas_err**2
    return np.sum(chi_arr)
    
def chi_for_min(x,mixing,gamma,Z,RGQ_meas,RGQ_meas_err):
    print(x)
    NHe = 10**x[0]
    NLi = 10**x[1]
    chi = chi_sq_RGQ(NHe,NLi,x[2],mixing,gamma,Z,RGQ_meas,RGQ_meas_err)
    print(chi)
    return chi

def solve_RGQ(Z,mixing,gamma,RGQ_meas,RGQ_meas_err):
    res = minimize(chi_for_min,x0=[18,17,100],args=(mixing,gamma,Z,RGQ_meas,RGQ_meas_err),bounds=[(15.1,19.9),(12.1,19.9),(50.1,399.9)])
    return res
