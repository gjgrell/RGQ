import numpy as np
from scipy.integrate import simps
from scipy import interpolate
from scipy.interpolate import interpn
from scipy import special
import sys
import pandas as pd
import matplotlib.pyplot as plt

from line_params import *
from RGQ import *

NHe_range = np.logspace(17,20,30)
NLi_range = np.logspace(16,19,30)
v_range = np.linspace(50,400,30)
phi_range = np.linspace(0, 1e5, 30)
ne_range = np.linspace(1, 1e12, 30)
mixing_range = np.linspace(0, 1, 30)

NLi_range_zoom = np.logspace(16.69,17.64, 30)


NHe = 1e18 #cm-2
NLi = 1e17 #cm-2
v = 150 #cm/s
phi = 0 #s-1
nele = 1 #cm-3
mixing = 1

R_O = []
G_O = []
Q_O = []

R_Si = []
G_Si = []
Q_Si = []

R_O_zoom = []
G_O_zoom = []
Q_O_zoom = []

R_Si_zoom = []
G_Si_zoom = []
Q_Si_zoom = []

for i in range(len(NLi_range)):

	Rs_O = R_analytic('O', NLi_range[i], NHe, v, phi, nele, mixing)
	Gs_O = G_analytic('O', NLi_range[i], NHe, v, phi, nele, mixing)
	Qs_O = Q_analytic('O', NLi_range[i], NHe, v, phi, nele)
	
	Rs_Si = R_analytic('Si', NLi_range[i], NHe, v, phi, nele, mixing)
	Gs_Si = G_analytic('Si', NLi_range[i], NHe, v, phi, nele, mixing)
	Qs_Si = Q_analytic('Si', NLi_range[i], NHe, v, phi, nele)
	
	Rs_O_zoom = R_analytic('O', NLi_range_zoom[i], NHe, v, phi, nele, mixing)
	Gs_O_zoom = G_analytic('O', NLi_range_zoom[i], NHe, v, phi, nele, mixing)
	Qs_O_zoom = Q_analytic('O', NLi_range_zoom[i], NHe, v, phi, nele)
	
	Rs_Si_zoom = R_analytic('Si', NLi_range_zoom[i], NHe, v, phi, nele, mixing)
	Gs_Si_zoom = G_analytic('Si', NLi_range_zoom[i], NHe, v, phi, nele, mixing)
	Qs_Si_zoom = Q_analytic('Si', NLi_range_zoom[i], NHe, v, phi, nele)
# 	
# 	Rs_O = R_analytic('O', NLi_range[i], NHe_range[i], v, phi, nele, mixing)
# 	Gs_O = G_analytic('O', NLi_range[i], NHe_range[i], v, phi, nele, mixing)
# 	Qs_O = Q_analytic('O', NLi_range[i], NHe_range[i], v, phi, nele)
# 	
# 	Rs_Si = R_analytic('Si', NLi_range[i], NHe_range[i], v, phi, nele, mixing)
# 	Gs_Si = G_analytic('Si', NLi_range[i], NHe_range[i], v, phi, nele, mixing)
# 	Qs_Si = Q_analytic('Si', NLi_range[i], NHe_range[i], v, phi, nele)

# 	Rs_O = R_analytic('O', NLi, NHe, v_range[i], phi, nele, mixing)
# 	Gs_O = G_analytic('O', NLi, NHe, v_range[i], phi, nele, mixing)
# 	Qs_O = Q_analytic('O', NLi, NHe, v_range[i], phi, nele)
# 	
# 	Rs_Si = R_analytic('Si', NLi, NHe, v_range[i], phi, nele, mixing)
# 	Gs_Si = G_analytic('Si', NLi, NHe, v_range[i], phi, nele, mixing)
# 	Qs_Si = Q_analytic('Si', NLi, NHe, v_range[i], phi, nele)

# 	Rs = R_analytic('Si', NLi, NHe, v, phi_range[i], nele, mixing)
# 	Gs = G_analytic('Si', NLi, NHe, v, phi_range[i], nele, mixing)
# 	Qs = Q_analytic('Si', NLi, NHe, v, phi_range[i], nele)

# 	Rs = R_analytic('O', NLi, NHe, v, phi, ne_range[i], mixing)
# 	Gs = G_analytic('O', NLi, NHe, v, phi, ne_range[i], mixing)
# 	Qs = Q_analytic('O', NLi, NHe, v, phi, ne_range[i])

	R_O.append(Rs_O)
	G_O.append(Gs_O)
	Q_O.append(Qs_O)

	R_Si.append(Rs_Si)
	G_Si.append(Gs_Si)
	Q_Si.append(Qs_Si)
	
	R_O_zoom.append(Rs_O_zoom)
	G_O_zoom.append(Gs_O_zoom)
	Q_O_zoom.append(Qs_O_zoom)

	R_Si_zoom.append(Rs_Si_zoom)
	G_Si_zoom.append(Gs_Si_zoom)
	Q_Si_zoom.append(Qs_Si_zoom)

plt.figure(figsize=(16,10))

plt.plot(NLi_range / NHe, R_O, color = 'b', linewidth=3.0, label =r'$\mathcal{R}$ (O)')
plt.plot(NLi_range / NHe, G_O, color='g', linewidth=3.0, linestyle='--',label =r'$\mathcal{G}$ (O)')
plt.plot(NLi_range / NHe, Q_O, color='r', linewidth=3.0, linestyle='dotted', label =r'$\mathcal{Q}$ (O)')
plt.plot(NLi_range / NHe, R_Si, color = 'purple', linewidth=3.0, label =r'$\mathcal{R}$ (Si)')
plt.plot(NLi_range / NHe, G_Si, color='cyan', linewidth=3.0, linestyle='--',label =r'$\mathcal{G}$ (Si)')
plt.plot(NLi_range / NHe, Q_Si, color='orange', linewidth=3.0, linestyle='dotted', label =r'$\mathcal{Q}$ (Si)')

plt.xscale('log')
plt.xlim(7e-3, 7e1)
#plt.xlim(20, 500)
#plt.ylim(0.1,15)
plt.yscale('log')

#plt.xlabel(r"$N_{Li}$ (cm$^{2}$)",fontsize=26)

#plt.xlabel(r"$N_{He}$ (cm$^{2}$)",fontsize=30)
plt.xlabel(r"$N_{Li}$ / $N_{He}$",fontsize=30)
#plt.xlabel("v (cm/s)",fontsize=30)
#plt.xlabel(r"$\phi_{UV}$ ($s^{-1}$)",fontsize=30)
#plt.xlabel(r"$n_e$ ($cm^{-3}$)",fontsize=30)

plt.ylabel(r'Ratios', fontsize = 30)
plt.legend(loc = 'best', fontsize = 22)#, bbox_to_anchor=(1, 0.99))



plt.tick_params(axis='x', which='major', labelsize=28)
plt.tick_params(axis='x', which='minor', labelsize=28)
plt.tick_params(axis='y', which='major', labelsize=28)
plt.tick_params(axis='y', which='minor', labelsize=28)

 # location for the zoomed portion 
sub_axes = plt.axes([.2, .6, .25, .25]) 

# plot the zoomed portion
sub_axes.plot(NLi_range_zoom / NHe, R_O_zoom, color = 'b', linewidth=3.0)
#sub_axes.plot(NLi_range_zoom / NHe, G_O_zoom, color='g', linewidth=3.0, linestyle='--')
#sub_axes.plot(NLi_range_zoom / NHe, Q_O_zoom, color='r', linewidth=3.0, linestyle='dotted')
sub_axes.plot(NLi_range_zoom / NHe, R_Si_zoom, color = 'purple', linewidth=3.0)
#sub_axes.plot(NLi_range_zoom / NHe, G_Si_zoom, color='cyan', linewidth=3.0, linestyle='--')
#sub_axes.plot(NLi_range_zoom / NHe, Q_Si_zoom, color='orange', linewidth=3.0, linestyle='dotted')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(3, 15)
plt.xlabel(r"$N_{Li}$ / $N_{He}$",fontsize=22)
plt.ylabel(r'$\mathcal{R}$', fontsize = 22)
#sub_axes.legend(loc = 'best', fontsize = 22)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='x', which='minor', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.tick_params(axis='y', which='minor', labelsize=20)
#plt.yaxis.set_major_formatter(ScalarFormatter())
#plt.yticks([10])
# plt.plot(NHe_range, R_O, color = 'b', linewidth=3.0, label ='R (O)')
# plt.plot(NHe_range, G_O, color='g', linewidth=3.0, linestyle='--',label ='G (O)')
# plt.plot(NHe_range, Q_O, color='r', linewidth=3.0, linestyle='dotted', label ='Q (O)')
# plt.plot(NHe_range, R_Si, color = 'purple', linewidth=3.0, label ='R (Si)')
# plt.plot(NHe_range, G_Si, color='cyan', linewidth=3.0, linestyle='--',label ='G (Si)')
# plt.plot(NHe_range, Q_Si, color='orange', linewidth=3.0, linestyle='dotted', label ='Q (Si)')


# plt.plot(v_range, R_O, color = 'b', linewidth=3.0, label ='R (O)')
# plt.plot(v_range, G_O, color='g', linewidth=3.0, linestyle='--',label ='G (O)')
# plt.plot(v_range, Q_O, color='r', linewidth=3.0, linestyle='dotted', label ='Q (O)')
# plt.plot(v_range, R_Si, color = 'purple', linewidth=3.0, label ='R (Si)')
# plt.plot(v_range, G_Si, color='cyan', linewidth=3.0, linestyle='--',label ='G (Si)')
# plt.plot(v_range, Q_Si, color='orange', linewidth=3.0, linestyle='dotted', label ='Q (Si)')


# plt.plot(phi_range, R, color = 'b', linewidth=3.0, label ='R')
# plt.plot(phi_range, G, color='g', linewidth=3.0, linestyle='--',label ='G')
# plt.plot(phi_range, Q, color='r', linewidth=3.0, linestyle='dotted', label ='Q')

# plt.plot(ne_range, R, color = 'b', linewidth=3.0, label ='R')
# plt.plot(ne_range, G, color='g', linewidth=3.0, linestyle='--',label ='G')
# plt.plot(ne_range, Q, color='r', linewidth=3.0, linestyle='dotted', label ='Q')





plt.savefig('ratios_NLi_NHe_ratio_zoomR_lin_O_Si.eps')
plt.savefig('ratios_NLi_NHe_ratio_zoomR_lin_O_Si.png')