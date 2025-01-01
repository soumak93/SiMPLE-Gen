#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate optical depth for halo sightlines.

This script contains functions to compute the optical depth for
Lyman-alpha absorption in the sightlines of halos, leveraging
cosmological and astrophysical parameters.

Author: Soumak
Created: Jan 1, 2025
"""

from halo_sightlines import *
from scipy.interpolate import interp1d
from scipy import special
import numpy as np
import time
from astropy.constants import c, m_p, m_e, k_B
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

###############################################################################
# Optical Depth Calculation Functions
###############################################################################

def tau(x, z, n_HI, T, v_pec):
    """
    Calculate optical depth (tau) for Lyman-alpha absorption.

    Parameters
    ----------
    x : ndarray
        Proper distance array (in Mpc).
    z : ndarray
        Redshift array.
    n_HI : ndarray
        Neutral hydrogen number density array.
    T : ndarray
        Temperature array (in Kelvin).
    v_pec : ndarray
        Peculiar velocity array (in km/s).

    Returns
    -------
    tau_z : ndarray
        Optical depth as a function of redshift.
    """
    I          = 4.45e-18  # Intensity constant
    x          = u.Mpc.to(u.cm) * x / cosmo.h
    lambda_lu  = 1215.67e-8  # Lyman-alpha wavelength in cm
    nu_lu      = c.to(u.cm / u.s).value / lambda_lu
    gamma_ul   = 6.262e8  # Damping coefficient in Hz
    m_H        = (m_p + m_e).to(u.g).value

    # Compute Doppler width (b), spacing (dx), and term1
    b          = np.sqrt(2.0 * k_B.to(u.erg / u.K).value * T / m_H)
    dx         = np.mean(x[1:] - x[:-1])
    term1      = (c.to(u.cm / u.s).value * I / np.sqrt(np.pi)) * dx * (n_HI / (b * (1 + z)))

    # Expand dimensions for vectorized computation
    z_expanded = z[None, :, None]
    b_expanded = b[:, :, None]

    term2      = np.real(
        special.wofz(
            1j * gamma_ul * c.to(u.cm / u.s).value / (4 * np.pi * nu_lu * b_expanded)
            + c.to(u.cm / u.s).value * (z_expanded - z[None, None, :]) / (b_expanded * (1 + z[None, None, :]))
            + v_pec[:, :, None] * 1e5 / b_expanded
        )
    )

    tau_z      = np.sum(term1[:, :, None] * term2, axis=1)

    return tau_z

def tau_halo_sightline(file_dir, z_centre, N_bunch=8192):
    """
    Generate optical depth (tau) for halo sightlines.

    Parameters
    ----------
    file_dir : str
        Directory containing input data files.
    z_centre : float
        Central redshift.
    N_bunch : int, optional
        Number of halos to process in each batch. Default is 8192.

    Returns
    -------
    Saves computed tau data to files in the specified directory.
    """
    # Load and refine data
    x_sim   = np.load(file_dir + "/x_sim_grid_023_M9_5.npy")[(255 - 48):(255 + 48)]
    z       = np.load(file_dir + "/z_grid_023_M9_5.npy")[(255 - 48):(255 + 48)]

    x_sim_new = np.linspace(np.min(x_sim), np.max(x_sim), 2 * len(x_sim))
    z         = np.linspace(np.min(z), np.max(z), 2 * len(z))

    n_HI_halo = np.load(file_dir + "/n_HI_halo_grid_023_M9_5.npy")[:, (255 - 48):(255 + 48)]
    T_halo    = np.load(file_dir + "/T_halo_grid_023_M9_5.npy")[:, (255 - 48):(255 + 48)]
    v_pec_halo = np.load(file_dir + "/v_pec_halo_grid_023_M9_5.npy")[:, (255 - 48):(255 + 48)]
    halomass  = np.load(file_dir + "/halomass_grid_023_M9_5.npy")

    # Interpolate to refined grid
    n_HI_halo = interp1d(x_sim, n_HI_halo, kind='linear', axis=1)(x_sim_new)
    T_halo    = interp1d(x_sim, T_halo, kind='linear', axis=1)(x_sim_new)
    v_pec_halo = interp1d(x_sim, v_pec_halo, kind='linear', axis=1)(x_sim_new)

    i_center  = int(len(n_HI_halo[0, :]) / 2) - 1
    vpos_halos_final = np.arange(-500, 501, 2.0)

    F_halo_final = np.zeros((len(halomass), len(vpos_halos_final)))

    halo_n = N_bunch

    for sec in range(int(len(halomass) / halo_n) + 1):
        print(f"Progress: {100.0 * sec / int(len(halomass) / halo_n):.2f}%")

        if int((sec + 1) * halo_n) < len(halomass):
            halo_indices = np.arange(int(sec * halo_n), int((sec + 1) * halo_n), 1)
        else:
            halo_indices = np.arange(int(sec * halo_n), len(halomass), 1)

        n_HI_halo_sec   = n_HI_halo[halo_indices, :]
        T_halo_sec      = T_halo[halo_indices, :]
        v_pec_halo_sec  = v_pec_halo[halo_indices, :]

        # Apply mask to zero out elements above each index
        I    = (i_center * np.ones(len(halo_indices))).astype(np.int32)
        mask = np.arange(n_HI_halo_sec.shape[1]) > I[:, None]
        n_HI_halo_sec[mask] = 0.0

        start_time = time.time()
        print(f"tau (dim = {n_HI_halo_sec.shape}) calculation started...")
        F_halo_los = tau(x_sim, z, n_HI_halo_sec, T_halo_sec, v_pec_halo_sec)
        print(f"tau (dim = {F_halo_los.shape}) calculation finished.")
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

        z_repeats      = np.repeat(z[np.newaxis, :], repeats=len(halo_indices), axis=0)
        vpos_halos     = c.to(u.km / u.s).value * (z_repeats - z_centre) / (1 + z_centre)

        # Vectorized interpolation
        F_halo_interpolated = np.array([
            np.interp(vpos_halos_final, vpos_halos[i, :], F_halo_los[i, :], left=np.nan, right=np.nan)
            for i in range(len(halo_indices))
        ])

        F_halo_final[sec * halo_n:sec * halo_n + len(halo_indices), :] = F_halo_interpolated

        del n_HI_halo_sec, T_halo_sec, v_pec_halo_sec

    # Save results
    np.save(file_dir + "/tau_halo_M_9_5_grid_023_v2.npy", F_halo_final)
