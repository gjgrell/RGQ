import sys
import numpy as np

def get_ratios(Z, phi):

    F = 0
    B = 0
    Btilde_x = 0
    Btilde_y = 0
    K = 0
    R0 = 0
    phi_uv = phi
    
    zline = 0
    xline = 0
    yline = 0
    wline = 0

    x_to_zline = 0
    y_to_zline = 0

    target = open('~/facfiles/recomb_ratios/'+Z+'_10.0_8.0_0.0-he.ln','r')
    all_lines = target.readlines()
    for line in all_lines:
        line_dat = line.split()
        if(len(line_dat)<6):
            continue
        line_dat = np.array(line_dat).astype(float)
        if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==6):
            wline = line_dat[6]
        if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==1):
            zline = line_dat[6]
        if (Z == 'C'):
            if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==2):
                yline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1] ==1 and line_dat[2]==2):
                y_to_zline = line_dat[6]
        if (Z == 'Ar' or Z == 'Ca' or Z == 'Cr' or Z == 'Mn' or Z == 'Fe' or Z == 'Ni'):
            if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==5):
                xline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==3):
                yline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==5):
                x_to_zline = line_dat[6]
            if (line_dat[0] == 2 and line_dat[1] ==1 and line_dat[2]==3):
                y_to_zline = line_dat[6]

        else:
            if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==4):
                xline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1]==0 and line_dat[2]==3):
                yline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==4):
                x_to_zline = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==3):
                y_to_zline = line_dat[6]
                
    x_params = get_params_He(Z, 'x')
    y_params = get_params_He(Z, 'y')
    
    xz_params = get_params_He(Z, 'xz')
    yz_params = get_params_He(Z, 'yz')
    
    B = (1./3.) * (y_params[2] / (y_params[2] + yz_params)) + (5./9.) * (x_params[2] / (x_params[2] + xz_params))
    #print(B)
    
    Btilde_x = (5./9.) * (x_params[2] / (x_params[2] + xz_params))    
    Btilde_y = (1./3.) * (y_params[2] / (y_params[2] + yz_params)) 
    
    UV_x = (phi_uv) * ((5./9.) * (x_params[2] / (x_params[2] + xz_params)))
    UV_y = (phi_uv) * ((1./3.) * (y_params[2] / (y_params[2] + yz_params)))
    
    R0 = (zline) / (xline + yline)
    F = (zline * B / (xline + yline)) - 1 + B    
    K = B * wline / (xline + yline)
    
    return F, B, Btilde_x, Btilde_y, UV_x, UV_y, K, R0


def get_params_He(Z, ln):
    w_E = 0
    w_osc = 0
    w_A = 0
    x_E = 0
    x_osc = 0
    x_A = 0
    y_E = 0
    y_osc = 0
    y_A = 0
    z_E = 0
    z_osc = 0
    z_A = 0
    
    xz_A = 0
    yz_A = 0
    
    target = open("~/facfiles/rates/"+Z+"02a.tr",'r')
    all_lines = target.readlines()
    for line in all_lines:
        line_dat = line.split()
        if(len(line_dat)<6):
            continue
        line_dat = np.array(line_dat).astype(float)
        if (line_dat[0] ==6 and line_dat[2]==0):
            w_E = line_dat[4]
            w_osc = line_dat[5]
            w_A = line_dat[6]
        if (line_dat[0] ==1 and line_dat[2]==0):
            z_E = line_dat[4]
            z_osc = line_dat[5]
            z_A = line_dat[6]
        if (Z == 'C'):
            if (line_dat[0] ==2 and line_dat[2]==0):
                y_E = line_dat[4]
                y_osc = line_dat[5]
                y_A = line_dat[6]
            if (line_dat[0] ==2 and line_dat[1] ==2 and line_dat[2]==1 and line_dat[5]>1e-6):
                yz_A = line_dat[6]
        if (Z == 'Ar' or Z == 'Ca' or Z == 'Cr' or Z == 'Mn' or Z == 'Fe' or Z == 'Ni'):
            if (line_dat[0] ==5 and line_dat[2]==0):
                x_E = line_dat[4]
                x_osc = line_dat[5]
                x_A = line_dat[6]
            if (line_dat[0] ==3 and line_dat[2]==0):
                y_E = line_dat[4]
                y_osc = line_dat[5]
                y_A = line_dat[6]
            if (line_dat[0] ==5 and line_dat[2]==1 and line_dat[5]>1e-6):
                xz_A = line_dat[6]
            if (line_dat[0] == 3 and line_dat[1] ==2 and line_dat[2]==1 and line_dat[5]>1e-6):
                yz_A = line_dat[6]

        else:
            if (line_dat[0] ==4 and line_dat[2]==0):
                x_E = line_dat[4]
                x_osc = line_dat[5]
                x_A = line_dat[6]
            if (line_dat[0] ==3 and line_dat[2]==0):
                y_E = line_dat[4]
                y_osc = line_dat[5]
                y_A = line_dat[6]
            if (line_dat[0] ==4 and line_dat[2]==1 and line_dat[5]>1e-6):
                xz_A = line_dat[6]
            if (line_dat[0] == 3 and line_dat[1] ==2 and line_dat[2]==1 and line_dat[5]>1e-6):
                yz_A = line_dat[6]

    if ln == 'w':
        return w_E, w_osc, w_A
    if ln == 'x':
        return x_E, x_osc, x_A
    if ln == 'y':
        return y_E, y_osc, y_A
    if ln == 'z':
        return z_E, z_osc, z_A
    if ln == 'xz':
        return xz_A
    if ln == 'yz':
        return yz_A
        
def get_params_Li(Z,ln):
    q_E = 0
    q_osc = 0
    q_A = 0
    r_E = 0
    r_osc = 0
    r_A = 0
    s_E = 0
    s_osc = 0
    s_A = 0
    t_E = 0
    t_osc = 0
    t_A = 0
    target = open("~/facfiles/rates/"+Z+"03a.tr",'r')
    all_lines = target.readlines()
    for line in all_lines:
        line_dat = line.split()
        if(len(line_dat)<6):
            continue
        line_dat = np.array(line_dat).astype(float)
        if (line_dat[0] ==436 and line_dat[2]==0):
            q_E = line_dat[4]
            q_osc = line_dat[5]
            q_A = line_dat[6]
        if (line_dat[0] ==435 and line_dat[2]==0):
            r_E = line_dat[4]
            r_osc = line_dat[5]
            r_A = line_dat[6]
        if (Z == 'Ca'):
            if (line_dat[0] ==441 and line_dat[2]==0):
                s_E = line_dat[4]
                s_osc = line_dat[5]
                s_A = line_dat[6]
            if (line_dat[0] ==439 and line_dat[2]==0):
                t_E = line_dat[4]
                t_osc = line_dat[5]
                t_A = line_dat[6]
        elif (Z == 'Cr' or Z == 'Mn' or Z == 'Fe' or Z == 'Ni'):
            if (line_dat[0] ==440 and line_dat[2]==0):
                s_E = line_dat[4]
                s_osc = line_dat[5]
                s_A = line_dat[6]
            if (line_dat[0] ==438 and line_dat[2]==0):
                t_E = line_dat[4]
                t_osc = line_dat[5]
                t_A = line_dat[6]
        else:
            if (line_dat[0] ==441 and line_dat[2]==0):
                s_E = line_dat[4]
                s_osc = line_dat[5]
                s_A = line_dat[6]
            if (line_dat[0] ==440 and line_dat[2]==0):
                t_E = line_dat[4]
                t_osc = line_dat[5]
                t_A = line_dat[6]

    if ln == 'q':
        return q_E, q_osc, q_A
    if ln == 'r':
        return r_E, r_osc, r_A
    if ln == 's':
        return s_E, s_osc, s_A
    if ln == 't':
        return t_E, t_osc, t_A
        
def get_ai_rates(Z, ln):
    q_ai = 0
    r_ai = 0
    s_ai = 0
    t_ai = 0
    
    target = open("~/facfiles/rates/"+Z+"03a.ai",'r')
    all_lines = target.readlines()
    for line in all_lines:
        line_dat = line.split()
        if(len(line_dat)<7):
            continue
        line_dat = np.array(line_dat).astype(float)
        if (line_dat[0] ==436 and line_dat[2]==8):
            q_ai = line_dat[5]
        if (line_dat[0] ==435 and line_dat[2]==8):
            r_ai = line_dat[5]
        if (Z == 'Ca'):
            if (line_dat[0] ==441 and line_dat[2]==8):
            	s_ai = line_dat[5]
            if (line_dat[0] ==439 and line_dat[2]==8):
            	t_ai = line_dat[5]
        elif (Z == 'Cr' or Z == 'Mn' or Z == 'Fe' or Z == 'Ni'):
            if (line_dat[0] ==440 and line_dat[2]==8):
                s_ai = line_dat[5]
            if (line_dat[0] ==438 and line_dat[2]==8):
                t_ai = line_dat[5]
        else:
            if (line_dat[0] ==441 and line_dat[2]==8):
                s_ai = line_dat[5]
            if (line_dat[0] ==440 and line_dat[2]==8):
                t_ai = line_dat[5]

    if ln == 'q':
        return q_ai
    if ln == 'r':
        return r_ai
    if ln == 's':
        return s_ai
    if ln == 't':
        return t_ai
        
def get_rates(Z):

    c_xz = 0
    c_zx = 0
    
    c_yz = 0
    c_zy = 0

    target = open('~/facfiles/ce_rates/'+Z+'_10.0_8.0_0.0.rt','r')
    all_lines = target.readlines()
    for line in all_lines:
        line_dat = line.split()
#         if(len(line_dat)<6):
#             continue
        line_dat = np.array(line_dat).astype(float)
        if (Z == 'C'):
            if (line_dat[0] ==2 and line_dat[1] ==1 and line_dat[2]==2):
                c_yz = line_dat[3]
                c_zy = line_dat[4]
        if (Z == 'Ar' or Z == 'Ca' or Z == 'Cr' or Z == 'Mn' or Z == 'Fe' or Z == 'Ni'):
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==5):
                c_xz = line_dat[3]
                c_zx = line_dat[4]
            if (line_dat[0] == 2 and line_dat[1] ==1 and line_dat[2]==3):
                c_yz = line_dat[3]
                c_zy = line_dat[4]

        else:
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==4):
                c_xz = line_dat[3]
                c_zx = line_dat[4]
            if (line_dat[0] ==2 and line_dat[1]==1 and line_dat[2]==3):
                c_yz = line_dat[3]
                c_zy = line_dat[4]
                
    C_xz = c_xz * 1e-10
    C_zx = c_zx * 1e-10
    
    C_yz = c_yz * 1e-10
    C_zy = c_zy * 1e-10
    
    return C_zx, C_zy
