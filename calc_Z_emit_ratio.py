import numpy as np
import matplotlib.pyplot as plt
import sys
import calc_qr_w_ratio
import os


def get_q_r_w_ratio(Z):
    nli_len = 50
    nhe_len = 50
    v_len = 5
    NLI_space = np.logspace(12,20,nli_len)
    NHE_space = np.logspace(15,20,nhe_len)
    vel_space = np.linspace(50,400,v_len)
    if(os.path.exists("q_r_w_ratio_"+str(Z)+".npy")):
        q_r_w_ratio = np.load("q_r_w_ratio_"+str(Z)+".npy")
    else:
        q_r_w_ratio = np.zeros((nli_len,nhe_len,v_len))
        for Nli in NLI_space:
            for NHe in NHE_space:
                for vel in vel_space:
                    e,spec = calc_qr_w_ratio.calc_qr_w_ratio(Nli,NHe,vel,Z)
                    i = np.argwhere(NLI_space==Nli)
                    j = np.argwhere(NHE_space==NHe)
                    k = np.argwhere(vel_space==vel)
                    spec_mid = int(np.round(len(spec)/2))
                    q_r_w_ratio[i,j,k] = np.sum(spec[:spec_mid])/np.sum(spec[spec_mid:])

        np.save("q_r_w_ratio_"+str(Z)+".npy",q_r_w_ratio)
    
    return NLI_space,NHE_space,vel_space,q_r_w_ratio