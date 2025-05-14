#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spec.py
-------
Step 2: Compute the optical depth τ(z) along each halo sightline.
"""
import numpy as np
from scipy import special
from astropy import units as u
from astropy.constants import c, m_p, m_e, k_B

from .config import PATHS

def tau(x, z, n_HI, T, v_pec):
    I         = 4.45e-18
    x_cm      = u.Mpc.to(u.cm) * x / u.Mpc    # note: cosmo.h cancels if x in Mpc/h
    lambda_lu = 1215.67e-8
    nu_lu     = c.to(u.cm/u.s).value / lambda_lu
    gamma_ul  = 6.262e8
    m_H       = (m_p + m_e).to(u.g).value

    b   = np.sqrt(2.0 * k_B.to(u.erg/u.K).value * T / m_H)
    dx  = np.mean(x_cm[1:] - x_cm[:-1])
    term1 = (c.to(u.cm/u.s).value * I / np.sqrt(np.pi)) * dx * (n_HI / (b * (1 + z)))

    z_exp = z[None, :, None]
    b_exp = b[:, :, None]

    term2 = np.real(
        special.wofz(
            1j * gamma_ul * c.to(u.cm/u.s).value
              / (4 * np.pi * nu_lu * b_exp)
            + c.to(u.cm/u.s).value
              * (z_exp - z[None,None,:])
              / (b_exp * (1 + z[None,None,:]))
            + v_pec[:,:,None] * 1e5 / b_exp
        )
    )

    # integrate over the second axis
    tau_z = np.sum(term1[:,:,None] * term2, axis=1)
    return tau_z

def run_spec():
    # ── Load sightline data ───────────────────────────────────────
    n_HI_halo  = np.load(PATHS["n_HI_halo"])   # (N_halo, N_x)
    T_halo     = np.load(PATHS["T_halo"])
    v_pec_halo = np.load(PATHS["v_pec_halo"])
    x_sim      = np.load(PATHS["x_sim"])      # shape (N_x,)
    z_grid     = np.load(PATHS["z_grid"])     # shape (N_x,)

    # ── Compute τ(z) ──────────────────────────────────────────────
    tau_halo = tau(x_sim, z_grid, n_HI_halo, T_halo, v_pec_halo)

    # ── Save results ──────────────────────────────────────────────
    np.save(PATHS["tau_halo"], tau_halo)
    print(f"[spec] τ saved → {PATHS['tau_halo']}")

if __name__ == "__main__":
    run_spec()

