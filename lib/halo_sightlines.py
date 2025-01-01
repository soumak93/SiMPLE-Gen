#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for reading simulation data and generating halo sightlines.

Created on Wed Jan 1, 2025
@author: Soumak
"""

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import Planck18 as cosmo

def read_field(file_directory, n_grid):
    """
    Reads density, temperature, ionization fraction, and velocity fields 
    from specified files.

    Parameters:
        file_directory (str): Path to the directory containing the data files.
        n_grid         (int): Number of grid points along one axis of the 3D grid.

    Returns:
        tuple: Density, temperature, ionization fraction, and velocity fields.
    """
    file_density    = f"{file_directory}cube_D_2048_instantanne_17.dat"
    file_temp       = f"{file_directory}cubeT/cube_T_2048_instantanne_17.dat"
    file_ion_frac   = f"{file_directory}cubeX/cube_X_2048_instantanne_17.dat"
    file_velocity   = f"{file_directory}grid_017/velx_field.npy"

    Dbox            = np.fromfile(file_density, dtype=np.float64, count=n_grid**3).reshape((n_grid, n_grid, n_grid)).astype(np.float32)
    print("Density field read...")

    Tbox            = np.fromfile(file_temp, dtype=np.float64, count=n_grid**3).reshape((n_grid, n_grid, n_grid)).astype(np.float32)
    print("Temperature field read...")

    Xbox            = np.fromfile(file_ion_frac, dtype=np.float64, count=n_grid**3).reshape((n_grid, n_grid, n_grid))
    print("Ionization fraction field read...")

    Vbox            = np.load(file_velocity).astype(np.float32)
    print("Velocity field read...")

    return Dbox, Tbox, Xbox, Vbox

def read_halo_catalog(file_directory):
    """
    Reads halo catalog data.

    Parameters:
        file_directory (str): Path to the directory containing the halo catalog file.

    Returns:
        tuple: Number of halos, halo masses, and halo center-of-mass positions.
    """
    file_halo_catalog = f"{file_directory}halo_catalog_015.dat"

    with open(file_halo_catalog, 'rb') as f:
        nhalo      = np.fromfile(f, dtype=np.int32, count=1)
        halomass   = np.fromfile(f, dtype=np.float32, count=nhalo[0])  # Halo masses (Msun)
        halo_cm    = np.fromfile(f, dtype=np.float32, count=3 * nhalo[0]).reshape((nhalo[0], 3))  # Positions (Mpc/h)

    print("Halo catalog read...")

    return nhalo, halomass, halo_cm

def generate_halo_sightlines(
    Dbox, Tbox, Xbox, Vbox, nhalo, halomass, halo_cm, 
    halomass_cutoff, z_centre, BoxSize, Outfile_dir
):
    """
    Generates sightlines for halos and saves results to specified directory.

    Parameters:
        Dbox            (np.ndarray): Density field.
        Tbox            (np.ndarray): Temperature field.
        Xbox            (np.ndarray): Ionization fraction field.
        Vbox            (np.ndarray): Velocity field.
        nhalo           (np.ndarray): Number of halos.
        halomass        (np.ndarray): Halo masses.
        halo_cm         (np.ndarray): Halo positions (center of mass).
        halomass_cutoff (float)     : Minimum halo mass to include.
        z_centre        (float)     : Central redshift of the simulation.
        BoxSize         (float)     : Physical size of the simulation box (Mpc/h).
        Outfile_dir     (str)       : Directory to save the output data.

    Returns:
        None
    """
    Boxlen        = Dbox.shape[0]
    i_center      = Boxlen // 2
    x_sim         = np.linspace(0., BoxSize, Boxlen, endpoint=False)
    x             = x_sim * u.Mpc / cosmo.h

    H_z           = cosmo.H(z_centre)
    dx_dz         = c / H_z
    z             = np.zeros_like(x)
    z[0]          = z_centre - (x[i_center] - x[0]) / dx_dz

    for i in range(len(z) - 1):
        z[i + 1] = z[i] + (x[i + 1] - x[i]) / dx_dz

    wavelength    = (1 + z) * 1215.67 * u.Angstrom

    # Compute neutral hydrogen number density
    nHbox         = Dbox / (u.m.to(u.cm) / (1.0 + z_centre))**3
    nHIbox        = nHbox * (1 - Xbox)

    # Filter halos by mass cutoff
    valid_halos   = halomass > halomass_cutoff
    halo_cm       = halo_cm[valid_halos]
    halomass      = halomass[valid_halos]

    halo_pos_grid = (halo_cm * Boxlen / BoxSize).astype(int)
    n_HI_halo     = nHIbox[:, halo_pos_grid[:, 1], halo_pos_grid[:, 2]].T
    T_halo        = Tbox[:, halo_pos_grid[:, 1], halo_pos_grid[:, 2]].T
    v_pec_halo    = Vbox[:, halo_pos_grid[:, 1], halo_pos_grid[:, 2]].T

    # Shift data to center on each halo
    shifts        = i_center - halo_pos_grid[:, 0]
    n_HI_halo     = np.roll(n_HI_halo, shift=shifts, axis=1)
    T_halo        = np.roll(T_halo, shift=shifts, axis=1)
    v_pec_halo    = np.roll(v_pec_halo, shift=shifts, axis=1)

    # Truncate to a region around the center
    region_slice  = slice(int(i_center - Boxlen / 8), int(i_center + Boxlen / 8))
    n_HI_halo     = n_HI_halo[:, region_slice]
    T_halo        = T_halo[:, region_slice]
    v_pec_halo    = v_pec_halo[:, region_slice]
    z             = z[region_slice]
    x_sim         = x_sim[region_slice]

    print("Halo sightlines extracted...")

    # Save results
    np.save(f"{Outfile_dir}n_HI_halo.npy",    n_HI_halo)
    np.save(f"{Outfile_dir}T_halo.npy",       T_halo)
    np.save(f"{Outfile_dir}v_pec_halo.npy",   v_pec_halo)
    np.save(f"{Outfile_dir}halo_cm.npy",      halo_cm)
    np.save(f"{Outfile_dir}halo_mass.npy",    halomass)
    np.save(f"{Outfile_dir}x_sim.npy",        x_sim)
    np.save(f"{Outfile_dir}z.npy",            z)


