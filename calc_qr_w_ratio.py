import numpy as np
from code import interact
from scipy.special import voigt_profile
from scipy.integrate import cumulative_trapezoid
from line_params import *
import pandas as pd


seed = 1234
N=int(1e6)
R_max = 1e14
R0 = 1e13
c=3e10
#abundances Mg,Si,S,Ar,Ca,Cr,Mn,Fe,Ni
PL=-1
m_e  = 510998.9 / (3e10)**2 #electron mass [eV / c^2]
hbar = 6.582e-16 #reduced Planck constant (eV*s)
h = hbar*2*np.pi
e2 = (hbar * c) / 137 #eV*cm
element_id = {6:"C",7:"N",8:"O",10:"Ne",12:"Mg",14:"Si",16:"S",18:"Ar",20:"Ca",24:"Cr",25:"Mn",26:"Fe",28:"Ni"}
rnd = np.random.default_rng(seed=seed)
vcdf_size = 10000
r_cdf = np.zeros(vcdf_size)
q_cdf = np.zeros(vcdf_size)
w_cdf = np.zeros(vcdf_size)
E_array = np.zeros(vcdf_size)
data_NIST = np.loadtxt('NIST_energies_qrst_wxyz.txt')
columns = ['Z', 'q', 'r', 's', 't', 'w','x','y','z']
index = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
df = pd.DataFrame(data_NIST,index=index, columns=columns)

MAX_THETA = 30*(np.pi/180)
COS_MAX_THETA = np.cos(MAX_THETA)
def randomize_photons(phot_list,E_min,E_max):
    ph_num = phot_list.shape[0]
    if(PL == -1):
        E = E_min*(E_max/E_min)**rnd.random(ph_num)
    else:
        E = ((E_max**(PL+1)-E_min)*rnd.random(ph_num) + E_min)**(1/(1+PL))
    # interact(local=dict(globals(), **locals()))
    # cos_v_theta = rnd.random(ph_num)*(1-2/np.sqrt(5)) + 2/np.sqrt(5)
    cos_v_theta = rnd.random(ph_num)*(1-COS_MAX_THETA) + COS_MAX_THETA
    # interact(local=dict(globals(), **locals())) 
    v_phi = 2*np.pi*rnd.random(ph_num) - np.pi
    vx,vy,vz = get_xyz(1,cos_v_theta,v_phi)
    phot_list[:,0] = E
    # ph_list[:,1] = -1e4
    phot_list[:,1] = 0
    phot_list[:,2] = 0
    phot_list[:,3] = 0
    # phot_list[:,3] = -2*R0
    phot_list[:,4] = vx
    phot_list[:,5] = vy
    phot_list[:,6] = vz
    phot_list[:,7] = 0
    phot_list[:,8] = 0
    phot_list[:,9] = 0



def get_xyz(r,cos_theta,phi):
    x = r*np.sqrt(1-cos_theta**2)*np.cos(phi)
    y = r*np.sqrt(1-cos_theta**2)*np.sin(phi)
    z = r*cos_theta
    return x,y,z


# get the cross section for absorption
def get_abs_cs_line(active_list,line_energy,f,broad_vel,gamma):
    base_sig = (np.pi * e2 /( m_e*c)) * f
    prof_sig = line_energy*broad_vel/c
    E = active_list[:,0]
    profile = h*voigt_profile(E-line_energy,prof_sig,gamma)
    # profile = (h/(prof_sig*np.sqrt(2*np.pi)))*np.exp(-((E-line_energy)**2)/(2*prof_sig**2))
    # interact(local=dict(globals(), **locals())) 
    return profile*base_sig

# def get_abs_cs_Li(active_list,energy_qrw,f_qr,broad_vel):
#     abs_cs = get_abs_cs_line(active_list,energy_qrw[0],f_qr[0],broad_vel) + get_abs_cs_line(active_list,energy_qrw[1],f_qr[1],broad_vel)+1e-100
#     # interact(local=dict(globals(), **locals())) 
#     return abs_cs

def progress_photons(active_list,dist):
    x_new = active_list[:,1] + dist*active_list[:,4]
    y_new = active_list[:,2] + dist*active_list[:,5]
    z_new = active_list[:,3] + dist*active_list[:,6]
    active_list[:,1] = x_new
    active_list[:,2] = y_new
    active_list[:,3] = z_new

# def get_path_integration(active_list,NLi,NHe,energy_qrw,w_energy,f_qr,w_f,broad_vel):
#     #here we determine the geometry - Li-like sphere on Z axis
#     x = active_list[:,1]
#     y = active_list[:,2]
#     z = active_list[:,3]
#     r2 = x**2+y**2+z**2
#     vx = active_list[:,4]
#     vy = active_list[:,5]
#     vz = active_list[:,6]
#     vel_times_loc = x*vx+y*vy+z*vz
#     nli = NLi/R0
#     nhe = NHe/R0
#     vel_times_loc = x*vx+y*vy+z*vz
#     Discriminant = vel_times_loc*vel_times_loc + R0**2 - r2
#     # interact(local=dict(globals(), **locals())) 
#     intersect_a = np.zeros_like(Discriminant)
#     intersect_b = np.zeros_like(Discriminant)
#     intersect_a[Discriminant>0] = -np.sqrt(Discriminant[Discriminant>0]) - vel_times_loc[Discriminant>0]
#     intersect_b[Discriminant>0] = np.sqrt(Discriminant[Discriminant>0]) - vel_times_loc[Discriminant>0]
#     intersect_a[intersect_a<0] = 0
#     intersect_b[intersect_b<0] = 0
#     max_len = intersect_b-intersect_a
#     abs_li_q = get_abs_cs_line(active_list,energy_qrw[0],f_qr[0],broad_vel)
#     abs_li_r = get_abs_cs_line(active_list,energy_qrw[1],f_qr[1],broad_vel)
#     abs_he = get_abs_cs_line(active_list,w_energy,w_f,broad_vel)
#     tau_q = max_len*abs_li_q*nli
#     tau_r = max_len*abs_li_r*nli
#     tau_he = max_len*abs_he*nhe
#     tau = tau_he+tau_r + tau_q
#     interaction_prob = (1-np.exp(-tau))
#     randomize = rnd.random(len(active_list[:,0]))
#     interaction_occured = randomize<=interaction_prob
#     abs_part = np.nonzero(interaction_occured )
#     interaction_position = np.zeros_like(r2)
#     interaction_position[abs_part] = -np.log(1-randomize[abs_part])/(((abs_li_q[abs_part]+abs_li_r[abs_part])*nli + abs_he[abs_part]*nhe))
#     escape_part = np.nonzero(1-interaction_occured)
#     interaction_position[escape_part] = R_max*10
#     # interact(local=locals())
#     return abs_part,interaction_position,tau_q,tau_r,tau_he


def get_path_integration(active_list,NLi,NHe,energy_qrw,f_qr,w_f,broad_vel,gamma_qrw):
    #here we determine the geometry - Li-like cone on Z axis
    x = active_list[:,1]
    y = active_list[:,2]
    z = active_list[:,3]
    r2 = x**2+y**2+z**2
    r = np.sqrt(r2)
    vx = active_list[:,4]
    vy = active_list[:,5]
    vz = active_list[:,6]
    nli = NLi/R0
    nhe = NHe/R0
    vel_times_loc = x*vx+y*vy+z*vz
    # theta = np.zeros_like(r)
    # theta[r>0] = np.arccos(z[r>0]/r[r>0])
    # outcone = theta>=MAX_THETA
    Discriminant = (COS_MAX_THETA**4) * (vel_times_loc*vel_times_loc - r2) + (COS_MAX_THETA**2)*(r2*vz**2+z**2-2*vz*z*vel_times_loc)
    # interact(local=dict(globals(), **locals())) 
    intersect = np.zeros_like(Discriminant)
    md = (-np.sqrt(Discriminant[Discriminant>=0]) + (COS_MAX_THETA**2)*vel_times_loc[Discriminant>=0] - z[Discriminant>=0]*vz[Discriminant>=0])/(vz[Discriminant>=0]**2 - COS_MAX_THETA**2)
    pd = (np.sqrt(Discriminant[Discriminant>=0]) + (COS_MAX_THETA**2)*vel_times_loc[Discriminant>=0] - z[Discriminant>=0]*vz[Discriminant>=0])/(vz[Discriminant>=0]**2 - COS_MAX_THETA**2)
    sel_md = (md<pd) * (md>0) + (pd<0)*(md>0)
    sel_pd = (md>pd) * (pd>0) + (pd>0)*(md<0)
    intersect[(Discriminant>=0)*sel_md] = md[sel_md]
    intersect[(Discriminant>=0)*sel_pd] = pd[sel_pd]
    intersect[intersect<=0] = R0
    intersect[intersect>R0] = R0
    # intersect[outcone] = 0
    max_len = intersect
    # interact(local=dict(globals(), **locals())) 
    abs_li_q = get_abs_cs_line(active_list,energy_qrw[0],f_qr[0],broad_vel,gamma_qrw[0])
    abs_li_r = get_abs_cs_line(active_list,energy_qrw[1],f_qr[1],broad_vel,gamma_qrw[1])
    abs_he = get_abs_cs_line(active_list,energy_qrw[2],w_f,broad_vel,gamma_qrw[2])
    tau_q = max_len*abs_li_q*nli
    tau_r = max_len*abs_li_r*nli
    tau_he = max_len*abs_he*nhe
    tau = tau_he+tau_r + tau_q
    interaction_prob = (1-np.exp(-tau))
    randomize = rnd.random(len(active_list[:,0]))
    interaction_occured = randomize<=interaction_prob
    abs_part = np.nonzero(interaction_occured )
    interaction_position = np.zeros_like(r2)
    interaction_position[abs_part] = -np.log(1-randomize[abs_part])/(((abs_li_q[abs_part]+abs_li_r[abs_part])*nli + abs_he[abs_part]*nhe))
    escape_part = np.nonzero(1-interaction_occured)
    interaction_position[escape_part] = R_max*10
    # interact(local=locals())
    return abs_part,interaction_position,tau_q,tau_r,tau_he
    



def  abs_interaction(active_list,abs_part,broad_vel,tau_q,tau_r,tau_he,energy_qrw,f_qr,omega_qr,rnd,E_min,E_max):
    abs_ph_list = active_list[abs_part]
    randomize = rnd.random(abs_ph_list.shape[0])
    abs_breakdown = np.zeros((abs_ph_list.shape[0],3))
    # tau_q = get_abs_cs_line(abs_ph_list,energy_qrw[0],f_qr[0],broad_vel) * tau_li[abs_part]/get_abs_cs_Li(abs_ph_list,energy_qrw,f_qr,broad_vel)
    # interact(local=dict(globals(), **locals()))
    tau_total = tau_q+tau_r+tau_he
    abs_breakdown[:,0] = tau_q[abs_part]/tau_total[abs_part]
    abs_breakdown[:,1] = (tau_q[abs_part]+tau_r[abs_part])/(tau_total[abs_part])
    abs_breakdown[:,2] = 1 
    interaction_occured = abs_breakdown>randomize[:,np.newaxis]
    not_abs_yet = np.ones_like(interaction_occured[:,0])
    cdf = q_cdf
    for i in range(0,3):
        if(i==1):
            cdf = r_cdf
        elif(i==2):
            cdf = w_cdf
        emit_prob = omega_qr[i]
        interaction = not_abs_yet*interaction_occured[:,i]
        ph_num = np.sum(interaction)
        emit_randomization = rnd.random(ph_num)
        emit = (emit_randomization <= emit_prob)
        interact_list = abs_ph_list[np.nonzero(interaction)]
    #replace all fully absorbed photons
        # temp_list = np.zeros_like(interact_list[np.nonzero(np.logical_not(emit))])
        # randomize_photons(temp_list,E_min,E_max)
        # interact(local=dict(globals(), **locals()))
        # interact_list[np.nonzero(np.logical_not(emit))] = np.copy(temp_list)
        interact_list[np.nonzero(np.logical_not(emit))] = 0
        u_sample = rnd.random(np.sum(emit))
        # interact_list[emit,0] = rnd.normal(energy_qrw[i],energy_qrw[i]*broad_vel/c,np.sum(emit))
        interact_list[emit,0] = np.interp(u_sample,cdf,E_array)
        new_cos_v_theta = 2*rnd.random(np.sum(emit))-1
        new_v_phi = 2*np.pi*rnd.random(np.sum(emit)) - np.pi
        new_vx,new_vy,new_vz = get_xyz(1,new_cos_v_theta,new_v_phi)
        interact_list[emit,4] = new_vx
        interact_list[emit,5] = new_vy
        interact_list[emit,6] = new_vz
        interact_list[emit,7] = 1
        if(i==2):
            interact_list[emit,8] += 1
            interact_list[emit,9] = 1
        abs_ph_list[np.nonzero(interaction)] = interact_list
        not_abs_yet = not_abs_yet*np.logical_not(interaction_occured[:,i])
    active_list[abs_part] = abs_ph_list  
    return 



def matter_interaction(ph_list,broad_vel,NLi,NHe,energy_qrw,f_qr,w_f,omega_qr,E_min,E_max,gamma_qrw):
    active = np.nonzero(ph_list[:,0]>0)
    active_list = ph_list[active]
    # randomize = rnd.random(active_list.shape[0])
    # interact(local=dict(globals(), **locals())) 
    # get probablities from geometry. This is where we set the geometry
    abs_part,interaction_position,tau_q,tau_r,tau_he = get_path_integration(active_list,NLi,NHe,energy_qrw,f_qr,w_f,broad_vel,gamma_qrw)
    #set escaped photons progress by R_max
    progress_photons(active_list,interaction_position)
    #absorb photons and re-emit (maybe)
    abs_interaction(active_list,abs_part,broad_vel,tau_q,tau_r,tau_he,energy_qrw,f_qr,omega_qr,rnd,E_min,E_max)
    ph_list[active] = active_list


# check how many photons are still in radius R
def test_photons_in_matter(ph_list,ph_output_list,R_max,iteration,saved_photons):
    #if they're out of the sim range they escape
    r = np.sqrt(ph_list[:,1]**2+ph_list[:,2]**2+ph_list[:,3]**2)
    escaped = r>=R_max
    start_idx = saved_photons
    save = np.nonzero(escaped)
    remain  = np.nonzero(1-escaped)
    saved_photons += np.sum(escaped)
    ph_output_list[start_idx:saved_photons] = ph_list[save]
    ph_list[save] = 0
    new_list = ph_list[np.nonzero(ph_list[:,0])]
    # if all photons are w and we had more than 50 iterations, just stop and assume they all escape
    if(np.all(ph_list[ph_list[:,0]>0,9]==1) and iteration>=50):
        save = np.nonzero(ph_list[:,0])
        ph_output_list[save] = ph_list[save]
        ph_list[save] = 0
    remaining = np.sum(ph_list[:,0]>0)
    # print(len(new_list))
    return new_list,remaining,saved_photons



def calc_qr_w_ratio(NLi,NHe,vel,Z):
    #He-like w emitting
    element = element_id[Z]
    He_w_params = get_params_He(element, 'w')
    w_centroid_energy = df.loc[element,'w']
    w_osc_strength = He_w_params[1]
    w_lorentz_gamma = He_w_params[2] * hbar #eV
   
    Li_r_params = get_params_Li(element, 'r')
    Li_q_params = get_params_Li(element, 'q')
    
    fac_ai_rate_r = get_ai_rates(element, 'r')
    fac_omega_r = Li_r_params[2] / (Li_r_params[2] + fac_ai_rate_r)
    
    fac_ai_rate_q = get_ai_rates(element, 'q')
    fac_omega_q = Li_q_params[2] / (Li_q_params[2] + fac_ai_rate_q)

    r_centroid_energy = df.loc[element,'r']
    q_centroid_energy = df.loc[element,'q']

    r_osc_strength = Li_r_params[1]
    r_lorentz_gamma = Li_r_params[2] * hbar #eV
    q_osc_strength = Li_q_params[1]
    q_lorentz_gamma = Li_q_params[2] * hbar #eV

    gamma_qrw = [q_lorentz_gamma,r_lorentz_gamma,w_lorentz_gamma]
    f_qr = [q_osc_strength,r_osc_strength]
    energy_qrw = [q_centroid_energy,r_centroid_energy,w_centroid_energy]
    w_f = w_osc_strength
    omega_qr = [fac_omega_q,fac_omega_r,1]
    E_min = energy_qrw[1]*(0.98)
    E_max = energy_qrw[2]*(1.02)
    ph_list = np.zeros((N,10))
    ph_output_list = np.zeros((N,10))
    bin_edges = np.arange(E_min,E_max,0.1)
    E_range = (bin_edges[:-1] + bin_edges[1:])/2
    ph_list[:,:] = 0
    E_array[:] = np.linspace(E_min,E_max,num=vcdf_size)
    randomize_photons(ph_list,E_min,E_max)
    ph_output_list[:,:] = 0
    print("Now working on:\n")
    print("Li-like Column density: " +str(NLi) +"\n He-like Column density:" + str(NHe) + "\n Velocity: " +str(vel)+ "\n ---------- \n")
    photons_in_matter = N
    iteration=0
    broad_vel = vel*1e5
    saved_photons = 0
    r_cdf[:] = cumulative_trapezoid(voigt_profile(E_array-energy_qrw[1],energy_qrw[1]*broad_vel/c,r_lorentz_gamma),E_array,initial=0)
    q_cdf[:] = cumulative_trapezoid(voigt_profile(E_array-energy_qrw[0],energy_qrw[0]*broad_vel/c,q_lorentz_gamma),E_array,initial=0)
    w_cdf[:] = cumulative_trapezoid(voigt_profile(E_array-energy_qrw[2],energy_qrw[2]*broad_vel/c,w_lorentz_gamma),E_array,initial=0)

    while(photons_in_matter>0):
        # interact(local=dict(globals(), **locals())) 
        print("This is iteration: " + str(iteration))
        print("There are " +str(photons_in_matter) + " photons in the simulation.")
        matter_interaction(ph_list,broad_vel,NLi,NHe,energy_qrw,f_qr,w_f,omega_qr,E_min,E_max,gamma_qrw)
        ph_list,photons_in_matter,saved_photons = test_photons_in_matter(ph_list,ph_output_list,R_max,iteration,saved_photons)
        iteration+=1
        # interact(local=dict(globals(), **locals())) 
    qr_part = ph_output_list[(ph_output_list[:,7]==1),0]
    # qr_scat,scat_bins = np.histogram(ph_output_list[(ph_output_list[:,7]==1)*(ph_output_list[:,8]>=1),8],bins=np.linspace(0,100,101))
    qr_hist,bins_temp= np.histogram(qr_part,bins=bin_edges)
    # interact(local=locals())
    return E_range,qr_hist
