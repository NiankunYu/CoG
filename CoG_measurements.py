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
fix_Vc = False
if fix_range == False:
    V_start =None
    V_end = None

dd0 = hca.sp_read(fits_file)
cog = spectrum(dd0)
rst_cog = cog.class_cog(V_c0, mirror=mirror, mirror_notes=mirror_notes, V_mirror = Vc_mirror, mask=mask_notes, vel_l=vel_l, vel_r=vel_r, V_start = V_start, V_end = V_end, fix_range = fix_range, fix_Vc= fix_Vc)
dd_final = rst_cog["dd_final"]
V_start = rst_cog["V_starti"]
V_end = rst_cog["V_endi"]
hca.sp_write(dd_final, ascii_path+ascii_file)
stat_cog = cog.class_stat(V_start = V_start, V_end = V_end, fix_range = True, fix_Vc= fix_Vc)
cog.cog_plot(pdf_path+pdf_file, le_name, cz, w50, w20)
cog.rst_write(txt_path+txt_file, ga_name)
del cog

# ################################# read the catalogue 
# name_100 = ("AGCNr","Name","RA_HI","DEC_HI","RA_OC","DEC_OC","Vhelio","W50","sigW",\
#             "W20","HIflux","sigflux","SNR","RMS","Dist","sigDist","logMH","siglogMH","HIcode",\
#             "AGCNr_legend", "AGCNr_file", "redshift","RAdeg_HI","DECdeg_HI","RAdeg_OC","DECdeg_OC")
# file_100 = "Haynes+2018_table2_alfalfa_flags.csv"
# df_100 = pd.read_csv(file_100, names=name_100, skiprows=1, sep=",", dtype={"AGCNr":int,"Name":str,\
#             "RA_HI":str,"DEC_HI":str,"RA_OC":str,"DEC_OC":str,"Vhelio":float,"W50":float,"sigW":float,\
#             "W20":float,"HIflux":float,"sigflux":float,"SNR":float,"RMS":float,"Dist":float,"sigDist":float,"logMH":float,\
#             "siglogMH":float,"HIcode":str, "AGCNr_legend":str, "AGCNr_file":str, "redshift":float,"RAdeg_HI":float,"DECdeg_HI":float,"RAdeg_OC":float,"DECdeg_OC":float})
# agc_100 = df_100.AGCNr
# str_100 = df_100.AGCNr_file
# legd_100 = df_100.AGCNr_legend
# Vc_100 = df_100.Vhelio
# W50_100 = df_100.W50
# W20_100 = df_100.W20
# flux_100 = df_100.HIflux
# dist_100 = df_100.Dist
# snr_100 = df_100.SNR
# rms_100 = df_100.RMS
# logmhi_100 = df_100.logMH
# flags = df_100.HIcode

# ###################################################################################################
# ########## read the notes file ####################################################################
# ###################################################################################################
# ############### the file below is from /Users/yuniankun/PROJECT/paper2_alpha100/2sample/spectral_measurements/alfalfa100_csvOrder_flags_finalv4.csv
# ############### which sets the ranges to measure the spectra successfully
# names_fp = ("AGCNr","Name","legd_name", "V_start", "V_end", "fix_range", "fix_Vc")
# file_fp = "alfalfa100_fixParams1.csv"
# df_fp = pd.read_csv(file_fp, names = names_fp, skiprows=1, sep=",", dtype = {"AGCNr":str,"Name":str,"legd_name":str, "V_start":float, "V_end":float, "fix_range":bool, "fix_Vc":bool})
# V_start = df_fp["V_start"]
# V_end = df_fp["V_end"]
# fix_range = df_fp["fix_range"]
# fix_Vc = df_fp["fix_Vc"]

# path = "/Users/yuniankun/PROJECT/paper2_alpha100/spectra/a100.180315.fits/spectraFITS/"
# # path = "../spectraFITS/"
# # pdf_path = "./measurement/pdf/"
# # txt_path = "./measurement/txt/"
# # ascii_path = "./measurement/ascii/"
# pdf_path = "./measurement/"
# txt_path = "./measurement/"
# ascii_path = "./measurement/"
# i_min = 16188
# i_max = i_min+1
# for i in range(i_min, i_max):
#     ga_name = str_100[i]
#     le_name = legd_100[i].replace("~", " ")
#     fits_file = ga_name+".fits"
#     pdf_file = ga_name+"_cog.pdf"
#     txt_file = ga_name+"_rst.txt"
#     ascii_file = ga_name+"_sp.txt"
#     ######### use the central velocity of ALFALFA
#     V_c0 = Vc_100[i]
#     czi = Vc_100[i]
#     w50i = W50_100[i]
#     w20i = W20_100[i]
#     mirrori = False
#     mirror_notesi = False
#     Vc_mirrori = V_c0
#     mask_notesi = False
#     vel_li = None 
#     vel_ri = None
#     V_starti = V_start[i]
#     V_endi = V_end[i]
#     fix_rangei = bool(fix_range[i])
#     fix_Vci = bool(fix_Vc[i])
#     if fix_rangei==False:
#         V_starti =None
#         V_endi = None
#     # if  (os.path.isfile(pdf_path+pdf_file)) | ("3" in flags[i]) | ("a" in flags[i]) | ("4" in flags[i]):
#     if  ("3" in flags[i]) | ("a" in flags[i]) | ("4" in flags[i]):
#         pass
#     else:
#         print(i, ga_name, czi, mirrori, mirror_notesi, mask_notesi, vel_li, vel_ri, V_starti, V_endi)
#         dd0 = hca.sp_read(path+fits_file)
#         cog = spectrum(dd0)
#         rst_cog = cog.class_cog(V_c0, mirror=mirrori, mirror_notes=mirror_notesi, V_mirror = Vc_mirrori, mask=mask_notesi, vel_l=vel_li, vel_r=vel_ri, V_start = V_starti, V_end = V_endi, fix_range = fix_rangei, fix_Vc= fix_Vci)
#         dd_final = rst_cog["dd_final"]
#         V_starti = rst_cog["V_starti"]
#         V_endi = rst_cog["V_endi"]
#         hca.sp_write(dd_final, ascii_path+ascii_file)
#         stat_cog = cog.class_stat(V_start = V_starti, V_end = V_endi, fix_range = True, fix_Vc= fix_Vci)
#         cog.cog_plot(pdf_path+pdf_file, le_name, czi, w50i, w20i)
#         cog.rst_write(txt_path+txt_file, ga_name)
#         del cog




