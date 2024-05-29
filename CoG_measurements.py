#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:10:53 2022
@author: Niankun Yu, email --  niankunyu@bao.ac.cn, niankun@pku.edu.cn
measure the spectrum using the curve-of-growth method, to derive the central velocity, total flux, 
line width, profile asymmetry, and profile shape. 
If you use this code, please cite our papers:
https://ui.adsabs.harvard.edu/abs/2020ApJ...898..102Y/abstract
https://ui.adsabs.harvard.edu/abs/2022ApJS..261...21Y/abstract
If you want to know more about our algorithm, please read the methodology in the above papers

Edited by Niankun Yu @ 2024.05.28, publish the code in the github


measure the spectra
(0) input parameters
(1) measure the spectra using the curve-of-growth method
(2) run Monte Carlo to derive the statistical uncertainties
(3) plot the figures
(4) save the final measurements
"""

import pandas as pd
import numpy as np
import math
import astropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import io, os, scipy, ast

import CoG_func as hca
from CoG_func import spectrum
#%%
###################################################################################################
###################################################################################################
###################################################################################################
################# input the parameters
path0 = os.path.dirname(__file__)
fits_file = path0+"/A000015.fits"
######## the galaxy nname
ga_name = "A000015"
######## the legend name
le_name = "AGC 000015"
######## the output files
pdf_path = path0+"/measurement/"
txt_path = path0+"/measurement/"
ascii_path = path0+"/measurement/"
pdf_file = ga_name+"_cog.pdf"
txt_file = ga_name+"_rst.txt"
ascii_file = ga_name+"_sp.txt"
######### the optical velocity or H I central velocity, to check if the algorithm find the signal automatically
V_c0 = 11580
cz = 11580
######### the guessed line width, usually the line width of nearby galaxies is 100-500 km/s
######### we do not input the line width into the algorithm, only for plotting the figure
w50 = 300
w20 = 400
mirror = False
mirror_notes = False
Vc_mirror = V_c0
########## mask the RFI or companions, default: no
mask_notes = False
########## the mask range: [vel_li, vel_ri], [vel_lj, vel_rj]
vel_l = None 
vel_r = None
########## for some certain case, the algorithm cannot identify the signal, 
########## we need to fix the velocity range as [V_start, V_end]
V_start = None
V_end = None
fix_range = False
########## fix the H I central velocity as V_c0 or not
fix_Vc = False
if fix_range == False:
    V_start =None
    V_end = None

#%%
###################################################################################################
###################################################################################################
###################################################################################################
################# measure the spectrum using the curve-of-growth method
########### read the H I spectrum as a dataframe
dd0 = hca.sp_read(fits_file)
########### initiate a class to measure the spectrum
cog = spectrum(dd0)
########### use the curve-of-growth method to measure the spectrum and save the results in rst_cog
rst_cog = cog.class_cog(V_c0, mirror=mirror, mirror_notes=mirror_notes, V_mirror = Vc_mirror, mask=mask_notes, vel_l=vel_l, vel_r=vel_r, V_start = V_start, V_end = V_end, fix_range = fix_range, fix_Vc= fix_Vc)
dd_final = rst_cog["dd_final"]
V_start = rst_cog["V_starti"]
V_end = rst_cog["V_endi"]
########### write the final spectrum, only show the signal and part of the baseline
hca.sp_write(dd_final, ascii_path+ascii_file)
########### measure the statistical uncertainties of all measured parameters
stat_cog = cog.class_stat(V_start = V_start, V_end = V_end, fix_range = True, fix_Vc= fix_Vc)
########### plot the curve of growth for blue-shift, red-shift, and total
cog.cog_plot(pdf_path+pdf_file, le_name, cz, w50, w20)
########### write the results from the curve-of-growth measurements, including the statistical uncertainties 
cog.rst_write(txt_path+txt_file, ga_name)
del cog
