#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:22:00 2025

@author: soumak
"""

from halo_sightlines import *

###############################################################################
# Optical Depth
###############################################################################


def tau(x, z, n_HI, T, v_pec):
    I = 4.45 * (10.0 ** -18)
    x = u.Mpc.to(u.cm) * x / cosmo.h
    lambda_lu = 1215.67 * (10.0 ** -8)
    nu_lu = c.to(u.cm / u.s).value / lambda_lu
    gamma_ul = 6.262 * (10.0 ** 8)
    m_H = (m_p + m_e).to(u.g).value
    
    # Compute b, dx, and term1 across all samples
    b = np.sqrt(2.0 * k_B.to(u.erg / u.K).value * T / m_H) 
    dx = np.mean(x[1:] - x[:-1])
    term1 = (c.to(u.cm / u.s).value * I / np.sqrt(np.pi)) * dx * (n_HI / (b * (1 + z))) 
    
    z_expanded = z[None, :, None] 
    b_expanded = b[:, :, None]   

    term2 = np.real(
        special.wofz(
            1j * gamma_ul * c.to(u.cm / u.s).value / (4 * np.pi * nu_lu * b_expanded)
            + c.to(u.cm / u.s).value * (z_expanded - z[None,None,:]) / (b_expanded * (1 + z[None,None,:]))
            + v_pec[:, :, None] * 10.0 ** 5 / b_expanded
        )
    )  
    
    tau_z = np.sum(term1[:, :, None] * term2, axis=1)
    
    return tau_z

def tau_halo_sightline(file_dir,z_centre,N_bunch=8192):
    x_sim=np.load(file_dir+"/x_sim_grid_023_M9_5.npy")[(255-48):(255+48)]
    z=np.load(file_dir+"/z_grid_023_M9_5.npy")[(255-48):(255+48)]

    x_sim_new = np.linspace(np.min(x_sim),np.max(x_sim),2*len(x_sim))
    z = np.linspace(np.min(z),np.max(z),2*len(z))


    n_HI_halo=np.load(file_dir+"/n_HI_halo_grid_023_M9_5.npy")[:,(255-48):(255+48)]
    T_halo=np.load(file_dir+"/T_halo_grid_023_M9_5.npy")[:,(255-48):(255+48)]
    v_pec_halo=np.load(file_dir+"/v_pec_halo_grid_023_M9_5.npy")[:,(255-48):(255+48)]
    halomass=np.load(file_dir+"/halomass_grid_023_M9_5.npy")

    f_n_HI = interp1d(x_sim, n_HI_halo, kind='linear', axis=1)
    n_HI_halo = f_n_HI(x_sim_new)
    f_T = interp1d(x_sim, T_halo, kind='linear', axis=1)
    T_halo = f_T(x_sim_new)
    f_v_pec = interp1d(x_sim, v_pec_halo, kind='linear', axis=1)
    v_pec_halo = f_v_pec(x_sim_new)
    
    i_center = int(len(n_HI_halo[0,:])/2)-1
    
    vpos_halos_final = np.arange(-500,501,2.0)

    F_halo_final = np.zeros((len(halomass),len(vpos_halos_final))) #np.load("/home/soumak/My_Files/Lya-Emitters/TIFR/tau_halo_M_9_5_grid_023.npy")

    halo_n = N_bunch

    for sec in range(int(len(halomass)/halo_n)+1):
        print("Progress="+str(100.0*sec/int(len(halomass)/halo_n))+"%")
        
        
        if int((sec+1)*halo_n) < len(halomass):
            halo_indices = np.arange(int(sec*halo_n),int((sec+1)*halo_n),1)
        else:
            halo_indices = np.arange(int(sec*halo_n),len(halomass),1)
        
        
       
        
        n_HI_halo_sec = n_HI_halo[halo_indices,:]
        T_halo_sec = T_halo[halo_indices,:]
        v_pec_halo_sec = v_pec_halo[halo_indices,:]
        
        
        
        
        
        # Create a mask for zeroing out elements above each index
        I = (i_center*np.ones(len(halo_indices))).astype(np.int32)
        mask = np.arange(n_HI_halo_sec.shape[1]) > I[:, None]
        
        # Apply the mask to set elements above each index to zero
        n_HI_halo_sec[mask] = 0.0
        
        start_time = time.time()
        print("tau (dim = " + str (n_HI_halo_sec.shape)+") calc started...")
        F_halo_los = tau(x_sim,z,n_HI_halo_sec,T_halo_sec,v_pec_halo_sec)
        print("tau (dim = " + str (F_halo_los.shape)+") calc finished...")
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.5f} seconds")
        
        z_repeats = np.repeat(z[np.newaxis,:],repeats=len(halo_indices),axis=0)
        vpos_halos = c.to(u.km/u.s).value*(z_repeats - z_centre)/(1+z_centre)
        
        
        # Perform vectorized interpolation using np.interp for each halo
        F_halo_interpolated = np.array([
            np.interp(vpos_halos_final, vpos_halos[i, :], F_halo_los[i, :], left=np.nan, right=np.nan)
            for i in range(len(halo_indices))
        ])
        
        # Assign interpolated values to the final array
        F_halo_final[sec * halo_n:sec * halo_n + len(halo_indices), :] = F_halo_interpolated
        
        del n_HI_halo_sec, T_halo_sec, v_pec_halo_sec
        

      
        np.save(file_dir+"/tau_halo_M_9_5_grid_023_v2.npy",F_halo_final)
