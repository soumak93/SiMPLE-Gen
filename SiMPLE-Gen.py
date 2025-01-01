#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:36:53 2025

@author: soumak
"""

# main.py
import sys
sys.path.append('./lib')  # Add the lib folder to sys.path

import halo_sightlines, optical_depth

z_sim = 7.14
halomass_cutoff = 10.0**9.5

file_dir = "/user1/soumak/Myfiles/Lya-Emitters/TIFR/"

Dbox, Tbox, Xbox, Vbox = halo_sightlines.field_read("/user1/sindhu/", n_grid=2048)

nhalo, halomass, halo_cm = halo_sightlines.halo_read("/user1/sindhu/halo_catalogs/halo_catalogs/")

halo_sightlines.halo_sightlines_gen(Dbox, Tbox, Xbox, Vbox, nhalo, halomass, halo_cm, halomass_cutoff = 10.0**9.5, z_centre = z_sim, BoxSize = 160.0, Outfile_dir=file_dir+"Files/")

optical_depth.tau_halo_sightline(file_dir+"Files/",z_sim,N_bunch=8192)
