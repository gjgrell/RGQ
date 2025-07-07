import numpy as np
from scipy.integrate import simps
from scipy import interpolate
from scipy.interpolate import interpn
from scipy import special
import sys
import pandas as pd

from line_params import *

#Constants
c = 3e10 #speed of light (cm/s)
m_e  = 510998.9 / (3e10)**2 #electron mass [eV / c^2]
m_p = 1.67e-24 #proton mass (grams)
#me = 9.11e-28 #grams
hbar = 6.582e-16 #Planck constant (eV*s)
h = 4.135e-15 #Planck constant (eV*s)
e2 = (hbar * c) / 137 #eV*cm


data_NIST = np.loadtxt('NIST_energies_qrst_wxyz.txt')
columns = ['Z', 'q', 'r', 's', 't', 'w','x','y','z']
index = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
df = pd.DataFrame(data_NIST,index=index, columns=columns)


# Power law function
def power_law(x, a, b):
    return a * np.power(x, -b)

#Voigt function    
def voigt(x, x0, sigma, gamma):
    return np.real(special.wofz((x - x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)


def R_analytic(Z, N_li, N_he, v, phi, nele, mixing, gamma):
    
    #____
    #Relevant ratios F, B, K for analytic R ratio calculation
    ratios = get_ratios(Z, phi)
    F = ratios[0] #Ratio of collisional population of z / (x + y)
    B = ratios[1] #Effective branching ratio
    Btilde_x = ratios[2] #Branching ratio for x to multiply by escape probability
    Btilde_y = ratios[3] #Branching ratio for y to multiply by escape probability
    UV_x = ratios[4]
    UV_y = ratios[5]
    K = ratios[6] #Ratio of radiative recombination rates of w / (x + y)
    R_0 = ratios[7]
    
    #____
    #R_pi_He, R_pi_Li
        
    #H = photoionization cross sections * ionizing radiation
    egrid_Li = []
    pi_sig_Li = []
    data_pi_Li = np.loadtxt('facfiles/RR_cs_Li/'+Z+'03_rr_cs_Li_K_edge.txt',skiprows=1,usecols=(0,3,4))
    
    for i in range(len(data_pi_Li)):
        egrid_Li.append(data_pi_Li[i][0])
        pi_sig_Li.append(data_pi_Li[i][2])

    y_Li = power_law(egrid_Li, 1, gamma)
    cs_Li = y_Li * pi_sig_Li #photoionization cross section
    R_pi_Li = 1e-20 * simps(cs_Li, egrid_Li) #ionization rate [FAC output units: 1e20 cm-2]
        
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0])
        pi_sig_He.append(data_pi_He[i][2])

    y_He = power_law(egrid_He, 1, gamma)
    cs_He = y_He * pi_sig_He
    R_pi_He = 1e-20 * simps(cs_He, egrid_He)
    
    #____
    #P_RAD, P_ESC
    
    #He-like x emitting
    He_x_params = get_params_He(Z, 'x')
    x_centroid_energy = df.loc[Z,'x']
    #x_centroid_energy = He_x_params[0]
    x_lorentz_gamma = He_x_params[2] * hbar #eV
    
    #He-like y emitting
    He_y_params = get_params_He(Z, 'y')
    y_centroid_energy = df.loc[Z, 'y']
    #y_centroid_energy = He_y_params[0]
    y_lorentz_gamma = He_y_params[2] * hbar #eV
    
    #Li-like s absorbing
    Li_s_params = get_params_Li(Z, 's')
    s_centroid_energy = df.loc[Z,'s']
    #s_centroid_energy = Li_s_params[0]
    fac_s_osc_strength = Li_s_params[1] #oscillator strength
    s_lorentz_gamma = Li_s_params[2] * hbar #eV
    
    #Li-like t absorbing
    Li_t_params = get_params_Li(Z, 't')
    t_centroid_energy = df.loc[Z,'t']
    #t_centroid_energy = Li_t_params[0]
    fac_t_osc_strength = Li_t_params[1] #oscillator strength
    t_lorentz_gamma = Li_t_params[2] * hbar #eV

    #Energy distribution
    e_lower = (y_centroid_energy - 300)
    e_upper = (y_centroid_energy + 300)
    e_grid = np.linspace(e_lower, e_upper, 100000) 

    #Line profiles
    sigma_em_x = (v * 1e5 / c) * x_centroid_energy
    sigma_em_y = (v * 1e5 / c) * y_centroid_energy
    
    sigma_em_s = (v * 1e5 / c) * s_centroid_energy
    sigma_em_t = (v * 1e5 / c) * t_centroid_energy

    phi_x = voigt(e_grid, x_centroid_energy, sigma_em_x, x_lorentz_gamma)
    phi_y = voigt(e_grid, y_centroid_energy, sigma_em_y, y_lorentz_gamma)
    phi_s = voigt(e_grid, s_centroid_energy, sigma_em_s, s_lorentz_gamma)
    phi_t = voigt(e_grid, t_centroid_energy, sigma_em_t, t_lorentz_gamma)

    #Absorption cross section for s
    sigma_abs = h * (np.pi * e2 / (m_e * c)) * (fac_s_osc_strength * phi_s + fac_t_osc_strength * phi_t)

    #omega - Radiative decay probabilities 
    fac_ai_rate_s = get_ai_rates(Z, 's')
    fac_omega_s = Li_s_params[2] / (Li_s_params[2] + fac_ai_rate_s)
    
    fac_ai_rate_t = get_ai_rates(Z, 't')
    fac_omega_t = Li_t_params[2] / (Li_t_params[2] + fac_ai_rate_t)
    
    #Probability distribution for photon absorption
    dPy = phi_y * (np.exp(-1 * sigma_abs * N_li))
    dPx = phi_x * (np.exp(-1 * sigma_abs * N_li))
    P_rad_y = mixing * (1 - simps(dPy, e_grid)) * (1 - (fac_omega_s + fac_omega_t))
    P_rad_x = mixing * (1 - simps(dPx, e_grid)) * (1 - (fac_omega_s + fac_omega_t)) 
    
    
    #____
    #A
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))
        
    #____
    #S
    
    #Ionization rate parameter
    S = N_li * R_pi_Li * K / (A * N_he * R_pi_He)
    
    #_____
    #R
    
    #Effective branching ratios weighted by escape probability
    Btilde_y_abs = Btilde_y * (1 - P_rad_y)
    Btilde_x_abs = Btilde_x * (1 - P_rad_x)
    Btilde = Btilde_y_abs + Btilde_x_abs
    
    He_z_params = get_params_He(Z, 'z')
    Azg = He_z_params[2]
    
    #Photoexcitation rates
    phi_c = Azg / (1 + F) #s-1
    
    #UV photoexcitation rate
    UVtilde_x_abs = UV_x * (1 - P_rad_x)
    UVtilde_y_abs = UV_y * (1 - P_rad_y)
    
    phi_UV = (1 / B) * (UV_x + UV_y) #s-1
    phi_tilde = (1 / Btilde) * (UVtilde_x_abs + UVtilde_y_abs) #s-1
    
    #Electron density
    n_e = nele #cm-3
    C_zi = get_rates(Z) #cm3/s #C = Azg / ((1 + F) * n_c)
    n_c = phi_c / C_zi[1] #cm-3

    #Substitutions for UV photoexcitation influence
    X = B * (phi_UV + n_e * C_zi[1])    
    Xtilde = Btilde * (phi_tilde + n_e * C_zi[1])

    R = (R_0 + S/B) / (((Btilde / B) * (1 + (phi_tilde / phi_c) + (n_e / n_c))) + S * Xtilde / (B * Azg))
    
    return R

    
def Q_analytic(Z, N_li, N_he, v, phi, nele, gamma):
        
    #____
    #H_He
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0])
        pi_sig_He.append(data_pi_He[i][2])

    y_He = power_law(egrid_He, 1, gamma)
    cs_He = y_He * pi_sig_He
    H_He = 1e-20 * simps(cs_He, egrid_He)
    
    #_____
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))
            
    #_____
    #L
    
    NLI_space = np.logspace(12,20,30)
    NHE_space = np.logspace(15,22,30)
    vel_space = np.linspace(50,400,5)

    qrw_ratio = np.load('q_r_w_ratio_'+Z+'.npy')
    Y = interpn((NLI_space,NHE_space,vel_space),qrw_ratio,[N_li,N_he,v])
    
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

    #Line profile
    sigma_em_w = (v * 1e5 / c) * w_centroid_energy
    voigt_w = voigt(e_grid_w, w_centroid_energy, sigma_em_w, w_lorentz_gamma)
    sigma_w = h * (np.pi * e2 / (m_e * c)) * fac_w_osc_strength * voigt_w
    tau_w = sigma_w * N_he

    dL_w = (1 - np.exp(-1*tau_w))
    L_w = simps(dL_w, e_grid)
    
    #Assumed parameters
    I_0 = 1 
    E_0 = 1 #eV

    I_w = I_0 * (w_centroid_energy / E_0)**(-1 * gamma)
    
    Q = L / (1 + (A * H_He * N_he * I_0 / (L_w * I_w)))
    
    return Q
    
def G_analytic(Z, N_li, N_he, v, phi, nele, mixing, gamma):
    
    #____
    #F, B, Btilde, K, R0
    
    #Relevant ratios F, B, K for analytic R ratio calculation
    ratios = get_ratios(Z, phi)
    F = ratios[0] #Ratio of collisional population of z / (x + y)
    B = ratios[1] #Effective branching ratio
    Btilde_x = ratios[2] #Branching ratio for x to multiply by escape probability
    Btilde_y = ratios[3] #Branching ratio for y to multiply by escape probability
    UV_x = ratios[4]
    UV_y = ratios[5]
    K = ratios[6] #Ratio of radiative recombination rates of w / (x + y)
    R_0 = ratios[7]
    
    #____
    #R_pi_Li, R_pi_He    
    #R_pi = photoionization cross sections * ionizing radiation
    egrid_Li = []
    pi_sig_Li = []
    data_pi_Li = np.loadtxt('facfiles/RR_cs_Li/'+Z+'03_rr_cs_Li_K_edge.txt',skiprows=1,usecols=(0,3,4))
    
    for i in range(len(data_pi_Li)):
        egrid_Li.append(data_pi_Li[i][0])
        pi_sig_Li.append(data_pi_Li[i][2])

    y_Li = power_law(egrid_Li, 1, gamma)
    cs_Li = y_Li * pi_sig_Li
    R_pi_Li = 1e-20 * simps(cs_Li, egrid_Li) #FAC output units: 1e20 cm-2
        
    egrid_He = []
    pi_sig_He = []
    data_pi_He = np.loadtxt('facfiles/RR_cs_He/'+Z+'02_rr_cs_He_K_edge.txt',skiprows=1,usecols=(0,3,4))

    for i in range(len(data_pi_He)):
        egrid_He.append(data_pi_He[i][0])
        pi_sig_He.append(data_pi_He[i][2])

    y_He = power_law(egrid_He, 1, gamma)
    cs_He = y_He * pi_sig_He
    R_pi_He = 1e-20 * simps(cs_He, egrid_He)
    
    #____
    #P_RAD, P_ESC
    
    #He-like x emitting
    He_x_params = get_params_He(Z, 'x')
    x_centroid_energy = df.loc[Z,'x']
    #x_centroid_energy = He_x_params[0]
    x_lorentz_gamma = He_x_params[2] * hbar #eV
    
    #He-like y emitting
    He_y_params = get_params_He(Z, 'y')
    y_centroid_energy = df.loc[Z, 'y']
    #y_centroid_energy = He_y_params[0]
    y_lorentz_gamma = He_y_params[2] * hbar #eV
    
    #Li-like s absorbing
    Li_s_params = get_params_Li(Z, 's')
    s_centroid_energy = df.loc[Z,'s']
    #s_centroid_energy = Li_s_params[0]
    fac_s_osc_strength = Li_s_params[1] #oscillator strength
    s_lorentz_gamma = Li_s_params[2] * hbar #eV
    
    #Li-like t absorbing
    Li_t_params = get_params_Li(Z, 't')
    t_centroid_energy = df.loc[Z,'t']
    #t_centroid_energy = Li_t_params[0]
    fac_t_osc_strength = Li_t_params[1] #oscillator strength
    t_lorentz_gamma = Li_t_params[2] * hbar #eV

    #Energy distribution
    e_lower = (y_centroid_energy - 300)
    e_upper = (y_centroid_energy + 300)
    e_grid = np.linspace(e_lower, e_upper, 100000) 

    #Line profiles
    sigma_em_x = (v * 1e5 / c) * x_centroid_energy
    sigma_em_y = (v * 1e5 / c) * y_centroid_energy
    
    sigma_em_s = (v * 1e5 / c) * s_centroid_energy
    sigma_em_t = (v * 1e5 / c) * t_centroid_energy

    phi_x = voigt(e_grid, x_centroid_energy, sigma_em_x, x_lorentz_gamma)
    phi_y = voigt(e_grid, y_centroid_energy, sigma_em_y, y_lorentz_gamma)
    phi_s = voigt(e_grid, s_centroid_energy, sigma_em_s, s_lorentz_gamma)
    phi_t = voigt(e_grid, t_centroid_energy, sigma_em_t, t_lorentz_gamma)

    #Absorption cross section for s
    sigma_abs = h * (np.pi * e2 / (m_e * c)) * (fac_s_osc_strength * phi_s + fac_t_osc_strength * phi_t)

    #omega - Radiative decay probabilities 
    fac_ai_rate_s = get_ai_rates(Z, 's')
    fac_omega_s = Li_s_params[2] / (Li_s_params[2] + fac_ai_rate_s)
    
    fac_ai_rate_t = get_ai_rates(Z, 't')
    fac_omega_t = Li_t_params[2] / (Li_t_params[2] + fac_ai_rate_t)
    
    #Probability distribution for photon absorption
    dPy = phi_y * (np.exp(-1 * sigma_abs * N_li)) 
    dPx = phi_x * (np.exp(-1 * sigma_abs * N_li))
    P_rad_y = mixing * ((1 - simps(dPy, e_grid)) * (1 - (fac_omega_s + fac_omega_t)))
    P_rad_x = mixing * (1 - simps(dPx, e_grid) * (1 - (fac_omega_s + fac_omega_t)))

    #____
    #A
    # A - Ratio of R rate for w line / R rate for all Hydrogen ions
    data_A = np.loadtxt('facfiles/A_calc/'+Z+'_10.0_6.0_0.0-he.ln')

    he_trans_to_ground = np.nonzero((data_A[:,0]==2)*(data_A[:,1]==0))
    w_loc = np.nonzero(data_A[he_trans_to_ground][:,2]==6)[0][0]

    A = (data_A[he_trans_to_ground][w_loc][6]/np.sum(data_A[he_trans_to_ground,6]))
        
    #print("A = " +str(A))
    
    #____
    #S
    
    #He-like w line
    He_w_params = get_params_He(Z, 'w')
    w_centroid_energy = df.loc[Z,'w']
    fac_w_osc_strength = He_w_params[1] 
    w_lorentz_gamma = He_w_params[2] * hbar #eV

    #Energy distribution
    e_w_lower = (w_centroid_energy - 300)
    e_w_upper = (w_centroid_energy + 300)
    e_grid_w = np.linspace(e_w_lower, e_w_upper, 100000) 

    #Line profile
    sigma_em_w = (v * 1e5 / c) * w_centroid_energy
    voigt_w = voigt(e_grid_w, w_centroid_energy, sigma_em_w, w_lorentz_gamma)
    sigma_w = h * (np.pi * e2 / (m_e * c)) * fac_w_osc_strength * voigt_w
    tau_w = sigma_w * N_he

    dL_w = (1 - np.exp(-1*tau_w))
    L_w = simps(dL_w, e_grid)
    
    #Assumed parameters
    I_0 = 1 
    E_0 = 1 #eV

    I_w = I_0 * (w_centroid_energy / E_0)**(-1 * gamma)
    
    S = N_li * R_pi_Li * K / (A * N_he * R_pi_He)
    
    #_____
    #G
    
    Btilde_y_abs = Btilde_y * (1 - P_rad_y)
    Btilde_x_abs = Btilde_x * (1 - P_rad_x)
    Btilde = Btilde_y_abs + Btilde_x_abs
    
    He_z_params = get_params_He(Z, 'z')
    Azg = He_z_params[2]
    
    #Photoexcitation rates
    phi_c = Azg / (1 + F) #s-1
    
    #UV photoexcitation rate
    UVtilde_x_abs = UV_x * (1 - P_rad_x)
    UVtilde_y_abs = UV_y * (1 - P_rad_y)
    
    phi_UV = (1 / B) * (UV_x + UV_y) #s-1
    phi_tilde = (1 / Btilde) * (UVtilde_x_abs + UVtilde_y_abs) #s-1
    
    #Electron density
    n_e = nele #cm-3
    C_zi = get_rates(Z) #cm3/s
    n_c = phi_c / C_zi[1] #cm-3

    #Substitutions for UV photoexcitation influence
    X = B * (phi_UV + n_e * C_zi[1])    
    Xtilde = Btilde * (phi_tilde + n_e * C_zi[1])
    
    G = ((1 + F - B + S) * ((Azg + Xtilde) / (Azg + X)) + Btilde) / (K + (I_w * L_w * K / (N_he * R_pi_He * A)))
    
    return G
    
