#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:09:28 2025

@author: soumak
"""


import numpy as np
from astropy import units as u
from astropy.constants import c,m_p,m_e,k_B
from astropy.cosmology import Planck18 as cosmo
import time




def field_read(file_directory, n_grid):

    file_path1 = file_directory+"cube_D_2048_instantanne_17.dat"
    data_type = np.float64  # Replace with the actual data type of your file, if known.
    Dbox = np.fromfile(file_path1, dtype=data_type,count=n_grid**3)
    Dbox = np.reshape(Dbox,(n_grid,n_grid,n_grid)).astype(np.float32)
    print("Density read...")
    
    file_path2 = file_directory+"cubeT/cube_T_2048_instantanne_17.dat"
    data_type = np.float64  # Replace with the actual data type of your file, if known.
    Tbox = np.fromfile(file_path2, dtype=data_type,count=n_grid**3)
    Tbox = np.reshape(Tbox,(n_grid,n_grid,n_grid)).astype(np.float32)
    print("Temperature read...")
    
    file_path3 = file_directory+"cubeX/cube_X_2048_instantanne_17.dat"
    data_type = np.float64  # Replace with the actual data type of your file, if known.
    Xbox = np.fromfile(file_path3, dtype=data_type,count=n_grid**3)
    Xbox = np.reshape(Xbox,(n_grid,n_grid,n_grid))
    print("Ionization fraction read...")
    
    Vbox = np.load(file_directory+"grid_017/velx_field.npy").astype(np.float32)
    print("Velocity read...")
    
    return Dbox, Tbox, Xbox, Vbox
    

def halo_read(file_directory):
    
    with open(file_directory+"halo_catalog_015.dat", 'rb') as f:
        nhalo = np.fromfile(f, dtype=np.int32, count=1)
        halomass = np.fromfile(f, dtype=np.float32, count=nhalo[0])  # Msun 
        halo_cm = np.fromfile(f, dtype=np.float32, count=3*nhalo[0]) # Mpc/h
        
    halo_cm = halo_cm.reshape((nhalo[0],3))
    
    print("halo catalog read...")
    
    return nhalo, halomass, halo_cm
    



def halo_sightlines_gen(Dbox, Tbox, Xbox, Vbox, nhalo, halomass, halo_cm, halomass_cutoff, z_centre, BoxSize, Outfile_dir):
    
    ###############################################################################
    # Length conversions
    ###############################################################################

    Boxlen  = len(Dbox[:,0,0])
    i_center = int(Boxlen/2)-1


    x_sim   = np.linspace(0.,BoxSize,Boxlen+1)[:-1] 

    x       = x_sim * u.Mpc / cosmo.h

    H_z     = cosmo.H(z_centre)
    dx_dz   = c / H_z 

    z       = np.zeros(len(x))
    z[0]    = z_centre - (x[int(Boxlen/2)]-x[0])/dx_dz

    for i in range(len(z)-1):
        z[i+1] = z[i] + (x[i+1]-x[i])/dx_dz
        
    wavelength = (1+z) * 1215.67
    
    
    ###############################################################################
    # nHI
    ###############################################################################

    nHbox = Dbox/(u.m.to(u.cm)/(1.0+z_centre))**3
    nHIbox = (nHbox * (1-Xbox) )
    del Xbox, Dbox
    
    ###############################################################################
    
    halo_cm = halo_cm[(halomass>halomass_cutoff),:]
    halomass= halomass[(halomass>halomass_cutoff)]

    halo_pos_grid = (halo_cm*Boxlen/BoxSize).astype(int)

    n_HI_halo = nHIbox[:,halo_pos_grid[:,1],halo_pos_grid[:,2]].T
    T_halo = Tbox[:,halo_pos_grid[:,1],halo_pos_grid[:,2]].T
    v_pec_halo = Vbox[:,halo_pos_grid[:,1],halo_pos_grid[:,2]].T

    n_HI_halo = np.roll(n_HI_halo,shift=(i_center-halo_pos_grid[:,0]),axis=1)
    T_halo = np.roll(T_halo,shift=(i_center-halo_pos_grid[:,0]),axis=1)
    v_pec_halo = np.roll(v_pec_halo,shift=(i_center-halo_pos_grid[:,0]),axis=1)



    n_HI_halo = n_HI_halo[:,int(i_center - Boxlen/8):int(i_center + Boxlen/8)]
    T_halo = T_halo[:,int(i_center - Boxlen/8):int(i_center + Boxlen/8)]
    v_pec_halo = v_pec_halo[:,int(i_center - Boxlen/8):int(i_center + Boxlen/8)]
    z = z[int(i_center - Boxlen/8):int(i_center + Boxlen/8)]
    x_sim = x_sim[int(i_center - Boxlen/8):int(i_center + Boxlen/8)]

    print("Halo sightlines extracted...")

    np.save(Outfile_dir+"n_HI_halo_grid_017_M9_5.npy",n_HI_halo)
    np.save(Outfile_dir+"T_halo_grid_017_M9_5.npy",T_halo)
    np.save(Outfile_dir+"v_pec_halo_grid_017_M9_5.npy",v_pec_halo)
    np.save(Outfile_dir+"halocm_grid_017_M9_5.npy",halo_cm)
    np.save(Outfile_dir+"halomass_grid_017_M9_5.npy",halomass)
    np.save(Outfile_dir+"x_sim_grid_017_M9_5.npy",x_sim)
    np.save(Outfile_dir+"z_grid_017_M9_5.npy",z)
    
    




