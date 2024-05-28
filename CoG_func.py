#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:52:01 2022
@author: Niankun Yu, email --  niankunyu@bao.ac.cn, niankun@pku.edu.cn
I finally reframe the structure of the curve-of-growth method.
measure the spectrum using the curve-of-growth method, to derive the central velocity, total flux, 
line width, profile asymmetry, and profile shape. 
If you use this code, please cite our papers:
https://ui.adsabs.harvard.edu/abs/2020ApJ...898..102Y/abstract
https://ui.adsabs.harvard.edu/abs/2022ApJS..261...21Y/abstract
If you want to know more about our algorithm, please read the methodology in the above papers
Edited by Niankun Yu @ 2024.05.28 -- some old functions does not work now, modify them

This code is based on
(a) the code in paper 1 
"/Users/yuniankun/PROJECT/paper1_CoG/3method/__HI_CoGclass_Version/HI_CoGclass_20201102.py", 
(b) the revision in paper 2
"/Users/yuniankun/PROJECT/paper2_alpha100/2sample/spectral_measurements/HI_CoGclass_alfalfa.py". 

###############################################################################################################
###############################################################################################################
###############################################################################################################
The logic flow in this version is:

(1) read the ALFALFA spectra as a dataframe

(2) get the channel spacing, drop nan values, reorder the spectra

(3) fit the distribution of negative and positive datapoints (V_c+-500 km/s)--> mean and sigma

(4) mask the spectra if necessary

(5) get the rough range of the spectra, which includes N channels
    (a) find small segments within V_c0+-500 km/s, where V_c0 is optical velocity of the object.
    detected segments: include at least three consecutive channels whose flux intensity is larger than 0.
    (b) calculate the integrated flux and mean flux density for each segment
    (c) find the segment with the largest integrated flux, and set is at the first detected signal. 
        This segment sets criteria for integrated flux and mean flux intensity.
    (d) find segments to two sides, whose integrated flux or mean flux density is larger than 0.5*criteria
    (e) the velocity separation of nearby segments should be smaller than 50 km/s
    
(6) calculate the median velocity and flux intensity-weighted central velocity

(7) extend 0.5N channels to each sides and build the curve of growth

(8) derive the total flux, line widths (V95, V85, V75, V25), asymmetry (A_F, A_C), and profile shape (C_V, K)

(9) run the Monte Carlo simulation to derive the statistical uncertainty

(10) plot the figures and save the final results

###############################################################################################################
###############################################################################################################
###############################################################################################################

set parameters as variables:
V_c+-500 km/s
three consecutive channels
0.5*criteria
extend 0.5N channels to each sides
"""

import math
from math import ceil
import numpy as np
import astropy
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
from astropy.io import ascii
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astroquery.ned import Ned
from scipy.optimize import curve_fit, bisect
from scipy.interpolate import interp1d
from scipy import stats
from numpy import mean, median, std
from itertools import groupby
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category = AstropyDeprecationWarning)

###############################################################################################
###############################################################################################
###############################################################################################
################ (1) read the spectra as dataframe
def sp_read(file):
    """
    read the raw fits file and return the dataframe containing the columns of velocity, flux and weight
    Edited by Niankun Yu @ 2024.05.28
    
    Parameters
    ----------
    file : str
        the fits file path and file name.
    Returns
    -------
    dd0: dataframe
        read the file and then save as the dataframe

    """
    hdulist1 = fits.open(file)
    hdu1 = hdulist1[1]
    data1 = hdu1.data
    # dtype=(numpy.record, [('VHELIO', '>f8'), ('FREQUENCY', '>f8'), ('FLUX', '>f8'), ('BASELINE', '>f8'), ('WEIGHT', '>f4')]))
    vhelio = data1.VHELIO
    frequency  = data1.FREQUENCY
    flux = data1.FLUX
    baseline = data1.BASELINE
    weight = data1.WEIGHT
    dd0 = pd.DataFrame({"velocity": vhelio, "flux": flux, "weight":weight}, dtype = float)
    return dd0

###############################################################################################
###############################################################################################
###############################################################################################
################ calculate the noise level by performing the Gaussian fitting to the flux distribution
def sp_sigma(dd0, V_c0, deltaV = None, weight_fileter = None, threshold=None, iters=None, V_start=None, V_end = None, vel_l = None, vel_r=None):
    """
    calculate the noise level of the spectrum
    We only use the range of V_c0+-deltaV
    if V_start and V_end are available, then we wil use the datapoints within the range [V_start, V_end],
    and further require their weights>0.5, and mirror the flux density distribution then use Gaussian function to 
    fit the mirrored distribution. Finally, we take the fitted sigma as the noise level of the spectrum
    
    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum.
    V_c0 : float
        the initical central velocity of the source. It should be the optical velocity if available
    deltaV : float, optional
        we will search for the signal within the range of [V_c0-deltaV, V_c0+deltaV]. The default is None.
    weight_fileter : float, optional
        the weight criteria of the spectra. The default is None.
    threshold : int, optional
        threshold for sigma-clipping. The default is None.
    iters : int, optional
        iteration times. The default is None.
    V_start : float, optional
        the minimum value of velocity while calculating the noise level of spectrum. The default is None.
    V_end : float, optional
        the maximum value of velocity while calculating the noise level of spectrum. The default is None.
    vel_l, vel_r: the list for mask
        
    Returns
    f_sigma : float, the noise level of the spectrum
    
    """
    if deltaV is None:
        deltaV = 500
    if weight_fileter is None:
        weight_fileter = 0.5
    if threshold is None:
        threshold=2
    if iters is None:
        iters=3
    if vel_l is not None:
    # if len(vel_l)!=0:
        dd0 = sp_mask(dd0, vel_l=vel_l, vel_r=vel_r)
    dd0 = dd0[(dd0["velocity"]>=V_c0-deltaV) & (dd0["velocity"]<=V_c0+deltaV)].reset_index(drop=True)
    vel_0 = dd0["velocity"]
    flux_0 = dd0["flux"]
    weight_0 = dd0["weight"]
    ############### (3) calculate the noise level of the spectrum (only use datapoints with weight>0.5)
    if (V_start is not None) and (V_end is not None):
        dd01 = dd0[(dd0["velocity"]>=V_start) & (dd0["velocity"]<=V_end)]
        dd_wgt = dd01[dd01["weight"]>weight_fileter].reset_index(drop=True)
    else:
        dd_wgt = dd0[dd0["weight"]>weight_fileter].reset_index(drop=True)
    flux_wgt = dd_wgt["flux"]
    weight_wgt = dd_wgt["weight"]
    ###### mirror (centered at 0) the distribution of flux density (flux<=0)
    flux_wgtN = flux_wgt[flux_wgt<0]
    flux_wgtNM = - flux_wgtN
    flux_wgtNM2 = np.concatenate((flux_wgtN, flux_wgtNM), axis=0)
    flux_wgtNM2SC = sigma_clip(flux_wgtNM2, sigma=threshold, maxiters=iters)
    ###### mirror (centered at 0) the distribution of flux density (flux>=0)
    flux_wgtP = flux_wgt[flux_wgt>=0]
    flux_wgtPM = - flux_wgtP
    flux_wgtPM2 = np.concatenate((flux_wgtP, flux_wgtPM), axis=0)
    flux_wgtPM2SC = sigma_clip(flux_wgtPM2, sigma=threshold, maxiters=iters)
    if np.all(abs(flux_wgtNM2SC)<=0.1) or np.all(abs(flux_wgtPM2SC)<=0.1):
        ##### for mock spectra, the Gaussian fitting may fail because all values are 0 at signal-free part
        f_mean=0
        f_sigma=0
    else:
        # get mean and standard deviation by mirroring the negative and positive distribution of the flux density
        f_meanN, sigmaN = stats.norm.fit(flux_wgtNM2SC)  
        f_meanP, sigmaP = stats.norm.fit(flux_wgtPM2SC)  
        f_sigma = min(sigmaN, sigmaP)
    return dd_wgt, f_sigma

###############################################################################################
###############################################################################################
###############################################################################################
################  drop datapoints with weight<=0.5 and invalid flux density, then let these flux densities =0
def sp_drop(dd0, weight_fileter = None, na_values = None):
    """
    drop flux densities of these useless channels (weight<weight_fileter or values = na_values)

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum.
    weight_fileter : float, optional
        the weight criteria of the spectra. The default is None.
    na_values : str, optional
        the expression of invalid values for flux density. The default is None.

    Returns
    -------
    dd1 : dataframe
        the dataframe after making some flux densities = 0

    """
    if weight_fileter is None:
        weight_fileter = 0.5
    if na_values is None:
        na_values = np.nan
    ############### for channels with spectral weight <=0.5, their flux densities = 0
    dd1 = dd0.copy()
    dd1["flux"][dd1["weight"]<=weight_fileter] = 0
    dd1["weight"][dd1["flux"]==na_values] = 0
    dd1["flux"][dd1["flux"]==na_values] = 0
    return dd1

###############################################################################################
###############################################################################################
###############################################################################################
################  order the spectrum by making the velocity increasing
def sp_order(dd0, ascending = True):
    """
    order the spectrum by making the velocity ascending

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum.
    ascending : True or False, optional
        the criteria for function: sort_values. The default is True.

    Returns
    -------
    dd1 : dataframe
        the dataframe with velocity ascending

    """
    if ascending is None:
        ascending = True
    dd1 = dd0.sort_values(by="velocity", ascending=ascending).reset_index(drop=True)
    return dd1

###############################################################################################
###############################################################################################
###############################################################################################
################  mirror the spectra, centered at one velocity
################  this function is not used in the final algorithm
def sp_mirror(dd0, V_mirror=None, mirror_notes=None, weight_fileter = None, na_values = None):
    """
    mirror the blue part the spectrum to get the signal at the red part if mirror_notes = blue
               red                                             blue                       red
               
    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum.
    V_mirror : float, optional
        the input spectra central, usually it is the optical velocity of the source. The default is None.
    mirror_notes : str, optional
        blue or red. It indicates the most reliable emission of the signal while performing mirroring. The default is None.
    weight_fileter : float, optional
        the weight criteria of the spectra. The default is None.
    na_values : str, optional
        the expression of invalid values for flux density. The default is None.
        
    Returns
    -------
    dd_mirror: dataframe
        The dataframe after mirror
    Vc_mirror: flaot
        The central velocity after mirror. 
        If we mirror the spectr and want to fix the central velocity while building the curve of grwoth,
        Then we have to set the central velocity as Vc_mirror.

    """
    vel = dd0["velocity"]
    flux = dd0["flux"]
    weight = dd0["weight"]
    flux_mirror = flux.copy()
    weight_mirror = weight.copy()
    if V_mirror is None:
        V_mirror = np.median(vel)
    if mirror_notes is None:
        mirror_notes = "red"
    if weight_fileter is None:
        weight_fileter = 0.5
    dd_wgt = dd0[dd0["weight"]>weight_fileter]
    if mirror_notes == "red":
        ###### find the index of the channel with velocity closest to V_mirror, and its weight should be higher than the critical weight
        dd_wgt_rm = dd_wgt[dd_wgt["velocity"]>=V_mirror]
        jj = dd_wgt_rm["velocity"].idxmin()
        Vc_mirror = dd_wgt_rm["velocity"][jj]
        min_len = min(len(flux)-jj-1, jj)
        for i in range(0, min_len):
            if weight[jj-i]<weight_fileter:
                flux_mirror[jj-i] = flux[jj+i]
                weight_mirror[jj-i] = max(weight[jj+i], 0.6)
    elif mirror_notes == "blue":
        dd_wgt_bm = dd_wgt[dd_wgt["velocity"]<=V_mirror]
        jj = dd_wgt_bm["velocity"].idxmax()
        Vc_mirror = dd_wgt_bm["velocity"][jj]
        min_len = min(len(flux)-jj-1, jj)
        for i in range(0, min_len):
            if weight[jj+i]<weight_fileter:
                flux_mirror[jj+i] = flux[jj-i]
                weight_mirror[jj+i] = max(weight[jj-i], 0.6)
    else:
        print("wrong input for the mirror_notes")   
    ######## the central velocity after mirror (remove some data-points with small weight)
    dd_mirror0 = dd0.copy()
    dd_mirror0["flux"] = flux_mirror
    dd_mirror0["weight"] = weight_mirror
    dd_mirror = sp_drop(dd_mirror0, weight_fileter =weight_fileter, na_values = na_values)
    return dd_mirror, Vc_mirror

###############################################################################################
###############################################################################################
###############################################################################################
################  mask the spectrum
def sp_mask(dd0, vel_l, vel_r, set_flux = None, set_weight=None, V_mirror=None, mirror_notes=None):
    """
    mask the RFI or Milky Way fluxes. If v in [vel_l, vel_r], and let the spectral weight = set_weight, flux = set_flux

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum.
    vel_l : array
        the minimum values of masks.
    vel_r : array
        the maximum values of masks.
    set_flux : float, optional
        the flux density for the masked region. The default is None.
    set_weight : float, optional
        the spectra weight for the masked refion. The default is None.
    V_mirror : float, optional
        the input spectra central, usually it is the optical velocity of the source. The default is None.
    mirror_notes : str, optional
        blue or red. It indicates the most reliable emission of the signal while performing mirroring. The default is None.
        
    Returns
    -------
    dd_mask : dataframe
        the dataframe after mask.

    """
    if set_flux is None:
        set_flux = 0
    if set_weight is None:
        set_weight = 0.2
    len_l = len(vel_l)
    len_r = len(vel_r)
    assert len_l == len_r
    ########## get the index of masked range
    idx_mask = []
    ########## if we mirror the spectrum before, then we have to deal with the mask too        
    if mirror_notes == "blue":
        if V_mirror is None:
            print("Please input the mirror velocity so that we could derive the mask properly.")
        ######## remove redundant mask
        if len_l !=0:
            for i in range(len(vel_l)):
                if vel_r[i]>V_mirror:
                    del vel_r[i]
                    del vel_l[i]
                else:
                    vel_l.append(vel_l[i]+2*(V_mirror-vel_l[i]))
                    vel_r.append(vel_r[i]+2*(V_mirror-vel_r[i]))
    elif mirror_notes == "red":
        if V_mirror is None:
            print("Please input the mirror velocity so that we could derive the mask properly.")
        ######## remove redundant mask
        if len_l !=0:
            for i in range(len(vel_l)):
                if vel_l[i]<V_mirror:
                    del vel_l[i]
                    del vel_r[i]
                else:
                    vel_l.append(vel_l[i]-2*(vel_l[i]-V_mirror))
                    vel_r.append(vel_r[i]-2*(vel_r[i]-V_mirror))
    len_l2 = len(vel_l)
    len_r2 = len(vel_r)           
    for i in range(len_l2):
        dd_maski = dd0[(dd0["velocity"]>vel_l[i]) & (dd0["velocity"]<vel_r[i])]
        idx_maski = dd_maski.index.values
        idx_mask.extend(idx_maski)    
    dd_mask = dd0.copy()
    dd_mask.loc[idx_mask, ("flux")] = set_flux
    dd_mask.loc[idx_mask, ("weight")] = set_weight
    return dd_mask

###############################################################################################
###############################################################################################
###############################################################################################
################  subtract the baseline
################  this function is not used in the final algorithm
def sp_baseline(dd0, vel_ls, vel_rs, deg=None):
    """
    subtract the baseline by polynominal fitting. For ALFALFA spectra, I did not use this function so far.

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum. 
        The dataframe should be masked before, otherwise the quality of polynominal fitting could be poor. 
    vel_l : float
        the minimum values of the signal.
    vel_r : float
        the maximum values of the signal.
    deg : int, optional
        the polynominal degree of polynominal fitting. The default is None.

    Returns
    -------
    dd_baseline: dataframe
        the dataframe after subtracting the baseline

    """
    if deg is None:
        deg = 1
    vel0 = dd0["velocity"]
    flux0 = dd0["flux"]
    dd_baseline = dd0.copy()
    ######### mask the signal, and perform polynominal fittting to the remaining datapoints
    ######### then subtract the baseline derived from polynominal fitting
    dd1 = dd0[(dd0["velocity"]<vel_ls) | (dd0["velocity"]>vel_ls)]
    vel1 = dd1["velocity"]
    flux1 = dd1["flux"]
    fit_func = np.poly1d(np.polyfit(vel1, flux1, deg))
    flux_baseline = fit_func(vel0)
    flux_fit = flux0 - flux_baseline
    ########## build the dataframe after subtracting the baseline
    dd_baseline["flux"] = flux_fit
    return dd_baseline

###############################################################################################
###############################################################################################
###############################################################################################
################  get the rough range of the signal   
def sp_range(dd0, V_c0, deltaV = None, V_start=None, V_end=None, f_ratio = None, m_ratio = None, V_diff = None, fix_range = False):
    """
    get the final signal range -- the range of the whole curve of growth
    Edited by Niankun Yu @ 2024.05.28

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum. 
    V_c0 : float
        the initical central velocity of the source. It should be the optical velocity if available
    deltaV : float, optional
        we will search for the signal within the range of [V_c0-deltaV, V_c0+deltaV]. The default is None.
    V_start : float, optional
        the minimum values of the velocity. The default is None.
    V_end : float, optional
        the maximum values of the velocity. The default is None.
        If V_strat and V_end are provided, we will search for the signal within the range of [V_start, V_end],
        instead of [V_c0-deltaV, V_c0+deltaV]
    f_ratio : float, optional
        the ratio of integrated fluxes between two patches. The default is None.
    m_ratio : float, optional
        the ratio of mean integrated fluxes between two patches. The default is None.
    V_diff : float, optional
        the minimum velocity difference of two nearby patches. 
        If A is detected as a signal, we will search for signal witinin [A_min-50, A_min] & [A_max, A_max+50]
        The default is None.

    Returns
    -------
    dd_signal : TYPE
        DESCRIPTION.
    idx_signal : TYPE
        DESCRIPTION.
    V_c : TYPE
        DESCRIPTION.

    """
    if deltaV is None:
        deltaV = 500
    V_min = float(V_c0 - deltaV)
    V_max = float(V_c0 + deltaV)
    if (V_start is not None) and (V_end is not None):
        dd_range0 = dd0[(dd0["velocity"]>V_start) & (dd0["velocity"]<V_end)]
    else:
        dd_range0 = dd0[(dd0["velocity"]>V_min) & (dd0["velocity"]<V_max)]
    dd_range = dd_range0.reset_index(drop=True)
    ################### if the signal range is fixed 
    if fix_range is True:
        dd_signal = dd_range.copy()
        return dd_range, dd_signal
    vel = dd_range["velocity"]
    flux = dd_range["flux"]
    len_tot = len(flux)
    ######### the index of positive flux density
    idx_pst = flux[flux>0].index.values
    diff_pst = np.diff(idx_pst)
    diff_pst_list = diff_pst.tolist()
    ######## group the index values if there are consecutive
    cst_pst = [len(list(group)) for key, group in groupby(diff_pst_list) if key==1]
    cst_pst_max = max(cst_pst)
    if f_ratio is None:
        f_ratio = 0.5
    if m_ratio is None:
        m_ratio = 0.5
    if V_diff is None:
        V_diff = 50
    idx_signal = []
    ################################## (1) find the patch with maximum integrated flux
    i = 0
    nn = 0
    f_meani = []
    f_sumi = []
    idx_sumi = []
    f_mean_max = 0
    f_sum_max = 0
    while i<=len(idx_pst)-2:
        if idx_pst[i+1]-idx_pst[i]>1:
            aa = 1
        else:
            ###################### !!!!!!! add the last channel in this part "sum(flux[ss[i]:ss[i+cst_ss[nn]]+1])", 
            ###################### rather than "sum(flux[ss[i]:ss[i+cst_ss[nn]+1]])"
            f_sum = sum(flux[idx_pst[i]:idx_pst[i+cst_pst[nn]]+1])
            f_mean = float(f_sum/(cst_pst[nn]+1))
            f_sumi.append(f_sum)
            f_meani.append(f_mean)
            idx_sumi.append(idx_pst[i:i+cst_pst[nn]+1])
            if f_mean>f_mean_max:
                f_mean_max = f_mean
            if f_sum>f_sum_max:
                f_sum_max = f_sum
                f_mean_criteria = m_ratio*f_mean
                f_sum_criteria = f_ratio*f_sum
                idx_sum_max = len(idx_sumi)-1
            aa = cst_pst[nn]
            nn = nn+1
        i = i+aa
    idx_signal.extend(idx_sumi[idx_sum_max])
    ################################ search to the blue-shift part 
    if idx_sum_max>1:
        for i in range(idx_sum_max):
            if (min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-1]])>V_diff):
                break
            elif (f_meani[idx_sum_max-i]>=f_mean_criteria) & (f_meani[idx_sum_max-i-1]>=f_mean_criteria) & (f_sumi[idx_sum_max-i-1]>=f_sum_criteria/5.0) & (min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-1]])<=V_diff):
                if min(vel[idx_signal])-max(vel[idx_sumi[idx_sum_max-i-1]])<=V_diff:########## this criteria is important, otherwise you may include segments which is far away from the central segment
                    idx_signal.extend(idx_sumi[idx_sum_max-i-1])
            elif (f_meani[idx_sum_max-i]>=f_mean_criteria) & (f_sumi[idx_sum_max-i-1]>=f_sum_criteria) & (min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-1]])<=V_diff):
                if min(vel[idx_signal])-max(vel[idx_sumi[idx_sum_max-i-1]])<=V_diff:
                    idx_signal.extend(idx_sumi[idx_sum_max-i-1])
            else:
                if idx_sum_max-i-1==0:
                    break
                elif idx_sum_max-i-1>=1:
                    for j in range(idx_sum_max-i-1):
                        if (f_meani[idx_sum_max-i]>=f_mean_criteria) & (f_meani[idx_sum_max-i-2-j]>=f_mean_criteria) & (f_sumi[idx_sum_max-i-2-j]>=f_sum_criteria/5.0) & (min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-2-j]])<=V_diff):
                            if min(vel[idx_signal])-max(vel[idx_sumi[idx_sum_max-i-2-j]])<=V_diff:
                                idx_signal.extend(idx_sumi[idx_sum_max-i-2-j])
                        elif (f_meani[idx_sum_max-i]>=f_mean_criteria) & (f_sumi[idx_sum_max-i-2-j]>=f_sum_criteria) & (min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-2-j]])<=V_diff):
                            if min(vel[idx_signal])-max(vel[idx_sumi[idx_sum_max-i-2-j]])<=V_diff:
                                idx_signal.extend(idx_sumi[idx_sum_max-i-2-j])
                        elif min(vel[idx_sumi[idx_sum_max-i]])-max(vel[idx_sumi[idx_sum_max-i-2-j]])>V_diff:
                            break
    ################################ search to the red-shift part      
    if idx_sum_max < len(idx_sumi):
        for i in range(idx_sum_max, len(idx_sumi)-1):
            if (f_meani[i]>=f_mean_criteria) & (f_meani[i+1]>=f_mean_criteria) & (f_sumi[i+1]>=f_sum_criteria/5.0) & (vel[min(idx_sumi[i+1])]-vel[max(idx_sumi[i])]<=V_diff):
                if min(vel[idx_sumi[i+1]]) - max(vel[idx_signal])<=V_diff:
                    idx_signal.extend(idx_sumi[i+1])
            elif (f_meani[i]>=f_mean_criteria) & (f_sumi[i+1]>=f_sum_criteria) & (vel[min(idx_sumi[i+1])]-vel[max(idx_sumi[i])]<=V_diff):
                if min(vel[idx_sumi[i+1]]) - max(vel[idx_signal])<=V_diff:
                    idx_signal.extend(idx_sumi[i+1])
            elif vel[min(idx_sumi[i+1])]-vel[max(idx_sumi[i])]>V_diff:
                break
            else:
                if i==len(idx_sumi)-1:
                    break
                elif i<len(idx_sumi)-1:
                    for j in range(i+1, len(idx_sumi)-1):
                        if (f_meani[i]>=f_mean_criteria) & (f_meani[j]>=f_mean_criteria) & (f_sumi[j]>=f_sum_criteria/5.0) & (vel[min(idx_sumi[j])]-vel[max(idx_sumi[i])]<=V_diff):
                            if min(vel[idx_sumi[j]]) - max(vel[idx_signal])<=V_diff:
                                idx_signal.extend(idx_sumi[j])
                        elif (f_meani[i]>=f_mean_criteria) & (f_sumi[j]>=f_sum_criteria) & (vel[min(idx_sumi[j])]-vel[max(idx_sumi[i])]<=V_diff):
                            if min(vel[idx_sumi[j]]) - max(vel[idx_signal])<=V_diff:
                                idx_signal.extend(idx_sumi[j])
                        elif vel[min(idx_sumi[j])]-vel[max(idx_sumi[i])]>V_diff:
                            break
    idx_signal.sort()   
    ################### if we did not find the signal, we will use the whole range of spectrum
    if len(idx_signal)==0:
        dd_signal = dd_range.copy()
        idx_signal = dd_signal.index.values
        return dd_range, dd_signal
    ################### to ensure there is enough datapoint to build the curve of growth
    elif len(idx_signal)<=4:
        idx_min = max(0, min(idx_signal)-4)
        idx_max = min(max(idx_signal)-4, len(dd_range)-1)
    else:
        idx_min = min(idx_signal)
        idx_max = max(idx_signal)
    ################ we do not reset index here so that we could compare dd_range and dd_signal by index later
    dd_signal = dd_range.iloc[idx_min:idx_max+1] 
    return dd_range, dd_signal

###############################################################################################
###############################################################################################
###############################################################################################
################ get the channel spacing of the spectrum
def sp_Vchan(dd0, decimal_n = None):
    """
    calculate the channel spacing of the spectrum

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum. 
        The dataframe should be dd_range, because the channel spacing should be measured just around the signal range
        In alfalfa fits file, the channel spacing is not a constant value
        I finally decide to take the median value of the channel spacing as the final channel spacing
    decimal_n : int, optional
        the number of decimal place for the channal spacing

    Returns
    -------
    the channel spacing: V_chan

    """
    if decimal_n is None:
        decimal_n = 1
    vel = dd0["velocity"]
    V_chan0 = np.diff(vel).round(decimal_n)
    V_chan = np.median(abs(V_chan0))
    return V_chan

###############################################################################################
###############################################################################################
###############################################################################################
################  calculate the central velocity of the spectra -- flux density weighted central velocity
def sp_Vc(dd0, sigma):
    """
    calculate the flux density weighted central velocity

    Parameters
    ----------
    dd0 : dataframe
        the dataframe with velocity, flux and weight of the spectrum. 
    sigma : float
        The noise level of the spectrum.

    Returns
    -------
    V_c : float
        The central velocity of the spectrum, either median velocity or flux density weighted central velocity.

    """
    vel_signal = dd0["velocity"]
    flux_signal = dd0["flux"]
    vel_min = min(vel_signal)
    vel_max = max(vel_signal)
    # dd_vc = dd0[dd0["velocity"]>sigma]  ##### we can not do this, otherwise the central velocity will systematically overestiamted (>20 km/s) at low S/N
    dd_vc = dd0.copy()
    vel_vc = dd_vc["velocity"]
    flux_vc = dd_vc["flux"]
    if len(vel_vc)<3:
        vel_wgt = sum(vel_signal*flux_signal)/sum(flux_signal)
        vel_med = np.median(vel_signal)
    else:
        vel_wgt = sum(vel_vc*flux_vc)/sum(flux_vc)
        vel_med = np.median(vel_vc)
    if (vel_wgt>=vel_max) or (vel_wgt<=vel_min):
        V_c = vel_med
    else:
        V_c = vel_wgt
    return V_c

###############################################################################################
###############################################################################################
###############################################################################################
################  interpolation function --> get the linewidth   
def sp_interp(vel, flux, flux_criteria, kind="slinear", xtol=None, rtol=None, maxiter=None):
    """
    get the interpolation of vel and flux, and derive the value of vel if flux = flux_criteria

    Parameters
    ----------
    vel : array
        The velocity array.
    flux : array
        The flux density array.
    flux_criteria : float
        The critical value of flux density.
    kind : str, optional
        The mathematical way of interpolation. The default is "slinear".
    xtol : float, optional
        The absolute tolerance parameter. The default is None.
    rtol : float, optional
        The relative tolerance parameter. The default is None.
    if absolute(xmin-xmax)<=xtol+rtol*absolute(xmax), biset will return the root
    maxiter : int, optional
        The maximum iteration time. The default is None.

    Returns
    -------
    vel_root : float
        The root of vel at flux_criteria.

    """
    if xtol is None:
        xtol=1.0
    if rtol is None:
        rtol=1e-4
    if maxiter is None:
        maxiter = 100
    interp_func = interp1d(vel, flux-flux_criteria, kind = kind)
    if (flux_criteria>max(flux)) or (flux_criteria<min(flux)):
        vel_root = 0
        warnings.warn("interpFunc: it is impossible to get right interpolation.")
        return vel_root
    vel_min = min(vel)
    vel_max = max(vel)
    ########## make sure that interp_func(vel_min) and interp_func(vel_max) have different signs
    while np.sign(interp_func(vel_min)) == np.sign(interp_func(vel_max)):
        vel_new = np.arange(vel_min, vel_max, 0.1)
        for i in range(len(vel_new)):
            sign1 = interp_func(vel_min)*interp_func(vel_new[i])
            sign2 = interp_func(vel_new[i])*interp_func(vel_max)
            if sign1<=0:
                vel_max = vel_new[i]
                break
            elif sign2<=0:
                vel_min = vel_new[i]
                break
    ########## find the root
    i_cycle = 0
    while i_cycle<200:
        if interp_func(vel_min) ==0 :
            root = vel_min
            return root
        elif interp_func(vel_max) ==0:
            root = vel_max
            return root
        elif interp_func(vel_min)*interp_func(vel_max) <0:
            root0 = bisect(interp_func, vel_min, vel_max, xtol=xtol, rtol=rtol, maxiter=maxiter)
            ########### confirm we find the smallest root
            root_check = np.arange(vel_min, root0, 0.1)
            if max(interp_func(root_check))<=0:
                root = root0
                return root
            else:
                ############# find the index of largest value in func(root_check)
                idx_lg, max_value = max(enumerate(interp_func(root_check)), key=lambda pair: pair[1])
                vel_max = root_check[idx_lg]
        i_cycle = i_cycle+1
    root = 5
    warnings.warn("We did not find the root by using the interpolation!")
    return root

###############################################################################################
###############################################################################################
###############################################################################################
################  interpolation function --> get the index of central velocity   
def sp_interpVc(vel, V_c):
    """
    

    Parameters
    ----------
    vel : array
        The velocity array.
    V_c : float
        The central velocity.

    Returns
    -------
    Vc_idx: int
       the index of the velocity closest to V_c

    """
    v_diff = vel - V_c
    v_diff_abs = abs(v_diff)
    v_diff_min = min(v_diff_abs)
    for i in range(len(v_diff_abs)):
        if v_diff_abs[i] == v_diff_min:
            Vc_idx = i
    return Vc_idx

###############################################################################################
###############################################################################################
###############################################################################################
################  build the curve of growth for the blue-shift, redshift and total spectrum      
def sp_cog(dd_range, dd_signal, V_c0, sigma, fix_Vc=True, n_extd = None, kind=None, V_chan=None, flux_frac = None, N_smo=None):
    """
    build the curve of growth: central velocity, total flux, line width, asymmetry, concentration, linear fitting to the curve of growth

    Parameters
    ----------
    dd_range : dataframe
        the dataframe with velocity range within [V_start, V_end] or [V_c0-deltaV, V_c0+deltaV].
    dd_signal : dataframe
        dd_signal should be part of dd_range or same as dd_range.
    V_c0 : float
        the central velocity, usually optical velocity.
    sigma: float
        noise level of the spectrum
    fix_Vc : bool, optional
        True or False. If fix_Vc is true, we build the curve of growth centered at V_c0. The default is True.
    n_extd : float, optional
        The extension times to two sides -- N channels in dd_signal, then (2*n_extd+1)*N channels in the dd_final. The default is None.
    kind : str, optional
        The interpolation way ("slinear"). The default is None.
    flux_frac : array, optional
        the flux fraction of line widths. The default is None.

    Returns
    -------
    results: lots of measurements from the spectra

    """
    if n_extd is None:
        n_extd = 0.5
    if kind is None:
        kind = "slinear"
    if flux_frac is None:
        flux_frac = [0.95, 0.85, 0.75, 0.25]
    ############ determine the central velocity, if the deviation of flux density weighted central velocity is 
    V_wgt = sp_Vc(dd_signal, sigma)  ######## flux density weighted central velocity
    if fix_Vc == True:
        if abs(V_wgt-V_c0)>200:
            V_c = V_c0
        else:
            V_c = V_wgt
    else:
        V_c = V_wgt
    ############ find the channel where the central velocity locate
    vel = dd_range["velocity"]
    flux = dd_range["flux"]
    min_idx0 = min(dd_range.index.values)
    max_idx0 = max(dd_range.index.values)
    idx_signal = dd_signal.index.values
    min_idx = min(idx_signal)
    max_idx = max(idx_signal)
    Vc_diff = vel - V_c
    Vc_diff_abs  = abs(Vc_diff)
    Vc_idx = Vc_diff_abs.idxmin()
    if Vc_idx - min_idx<=2:
        min_idx = max(Vc_idx-3, min_idx0)
    if max_idx - Vc_idx<=2:
        max_idx = min(Vc_idx+3, max_idx0)
    N_chan1 = max_idx-min_idx
    min_idx2 = int(max(min_idx0, min_idx-N_chan1*n_extd))
    max_idx2 = int(min(max_idx0, max_idx+N_chan1*n_extd))
    N_chan2 = max_idx2-min_idx2
    ############ the final dataframe used to build the curve of growth
    dd_final = dd_range.iloc[min_idx2:max_idx2].reset_index(drop=True)
    vel_final= dd_final["velocity"]
    flux_final = dd_final["flux"]
    ############ the velocity range of the spectra
    V_starti = min(vel_final)
    V_endi = max(vel_final)
    len_final = len(vel_final)
    Vc_diff_final = vel_final-V_c
    Vc_diff_final_abs = abs(Vc_diff_final)
    Vc_idx_final = Vc_diff_final_abs.idxmin()
    flux_Vc = flux_final[Vc_idx_final]
    ############ the rough signal channel
    len_b = Vc_idx - min_idx
    len_r = max_idx - Vc_idx
    ########### the extension to two sides
    len_ext_b = Vc_idx_final - len_b
    len_ext_r = len_final-1 - (Vc_idx_final+len_r)
    len_1 = min(len_b, len_r)
    len_2 = max(len_b, len_r)
    len_3 = min(len_ext_b, len_ext_r)
    len_4 = max(len_ext_b, len_ext_r)
    width_b = 0
    flux_b = 0
    width_r = 0
    flux_r =0
    width_t = 0
    flux_t =0
    cog_width_b = []
    cog_flux_b = []
    cog_width_r = []
    cog_flux_r = []
    cog_width_t = []
    cog_flux_t = []
    cog_width_b.append(width_b)
    cog_flux_b.append(flux_b)
    cog_width_r.append(width_r)
    cog_flux_r.append(flux_r)
    cog_width_t.append(width_t)
    cog_flux_t.append(flux_t)
    ########## deal with the channel where the central velocity locate
    width_b = width_b + V_c - (vel_final[Vc_idx_final]+vel_final[Vc_idx_final-1])/2.0
    flux_b = flux_b + flux_final[Vc_idx_final]*width_b
    width_r = width_r + (vel_final[Vc_idx_final+1]+vel_final[Vc_idx_final])/2.0-V_c
    flux_r = flux_r + flux_final[Vc_idx_final]*width_r
    cog_width_b.append(width_b)
    cog_flux_b.append(flux_b)
    cog_width_r.append(width_r)
    cog_flux_r.append(flux_r)
    ######### build the curve of growth for two sides -- the signal part
    for i in range(len_b-1):
        width_b = width_b + (vel_final[Vc_idx_final-i]-vel_final[Vc_idx_final-i-1])
        flux_b = flux_b + flux_final[Vc_idx_final-i-1]*(vel_final[Vc_idx_final-i]-vel_final[Vc_idx_final-i-1])
        cog_width_b.append(width_b)
        cog_flux_b.append(flux_b)
    for i in range(len_r-1):
        width_r = width_r + (vel_final[Vc_idx_final+i+1]-vel_final[Vc_idx_final+i])
        flux_r = flux_r + flux_final[Vc_idx_final+i+1]*(vel_final[Vc_idx_final+i+1]-vel_final[Vc_idx_final+i])
        cog_width_r.append(width_r)
        cog_flux_r.append(flux_r)
    ######## build the total curve of growth -- align the emission
    for i in range(len_1):
        width_t = width_t+(cog_width_r[i+1]-cog_width_r[i])+(cog_width_b[i+1]-cog_width_b[i])
        flux_t = flux_t+(cog_flux_r[i+1]-cog_flux_r[i])+(cog_flux_b[i+1]-cog_flux_b[i])
        cog_width_t.append(width_t)
        cog_flux_t.append(flux_t)
    if len_b>len_r:
        for i in range(len_1, len_2):
            width_t = width_t+(cog_width_b[i+1]-cog_width_b[i])
            flux_t = flux_t+(cog_flux_b[i+1]-cog_flux_b[i])
            cog_width_t.append(width_t)
            cog_flux_t.append(flux_t)
    elif len_b<len_r:
        for i in range(len_1, len_2):
            width_t = width_t+(cog_width_r[i+1]-cog_width_r[i])
            flux_t = flux_t+(cog_flux_r[i+1]-cog_flux_r[i])
            cog_width_t.append(width_t)
            cog_flux_t.append(flux_t)
    ######### build the curve of growth for two sides -- the baseline part
    for i in range(len_ext_b-1):
        width_b = width_b + (vel_final[Vc_idx_final-len_b-i]-vel_final[Vc_idx_final-len_b-i-1])
        flux_b = flux_b + flux_final[Vc_idx_final-len_b-i-1]*(vel_final[Vc_idx_final-len_b-i]-vel_final[Vc_idx_final-len_b-i-1])
        cog_width_b.append(width_b)
        cog_flux_b.append(flux_b)
    for i in range(len_ext_r-1):
        width_r = width_r + (vel_final[Vc_idx_final+len_r+i+1]-vel_final[Vc_idx_final+len_r+i])
        flux_r = flux_r + flux_final[Vc_idx_final+len_r+i+1]*(vel_final[Vc_idx_final+len_r+i+1]-vel_final[Vc_idx_final+len_r+i])
        cog_width_r.append(width_r)
        cog_flux_r.append(flux_r)
    ######## build the total curve of growth -- align the emission
    for i in range(len_3-1):
        width_t = width_t+(cog_width_r[len_r+i+1]-cog_width_r[len_r+i])+(cog_width_b[len_b+i+1]-cog_width_b[len_b+i])
        flux_t = flux_t+(cog_flux_r[len_r+i+1]-cog_flux_r[len_r+i])+(cog_flux_b[len_b+i+1]-cog_flux_b[len_b+i])
        cog_width_t.append(width_t)
        cog_flux_t.append(flux_t)
    if len_ext_b>len_ext_r:
        for i in range(len_3, len_4-2):
            width_t = width_t+(cog_width_b[len_b+i+1]-cog_width_b[len_b+i])
            flux_t = flux_t+(cog_flux_b[len_b+i+1]-cog_flux_b[len_b+i])
            cog_width_t.append(width_t)
            cog_flux_t.append(flux_t)
    elif len_ext_b<len_ext_r:
        for i in range(len_3, len_4-2):
            width_t = width_t+(cog_width_r[len_r+i+1]-cog_width_r[len_r+i])
            flux_t = flux_t+(cog_flux_r[len_r+i+1]-cog_flux_r[len_r+i])
            cog_width_t.append(width_t)
            cog_flux_t.append(flux_t)
    ######### calculate integrated flux at the flat part of the curve of growth
    if len_ext_b<=2:
        flux_b_tot = cog_flux_b[len_b]
    else:
        flux_b_tot = median(cog_flux_b[-len_ext_b:])
    if len_ext_r<=2:
        flux_r_tot = cog_flux_r[len_r]
    else:
        flux_r_tot = median(cog_flux_r[-len_ext_r:])
    if (len_ext_b<=2) & (len_ext_r<=2):
        flux_t_tot = cog_flux_t[len_1-1]
    else:
        flux_t_tot = median(cog_flux_t[-max(len_ext_b, len_ext_r):])
    ####### calculate the line width
    v1 = sp_interp(cog_width_t, cog_flux_t, flux_frac[0]*flux_t_tot, kind=kind)
    v2 = sp_interp(cog_width_t, cog_flux_t, flux_frac[1]*flux_t_tot, kind=kind)
    v3 = sp_interp(cog_width_t, cog_flux_t, flux_frac[2]*flux_t_tot, kind=kind)
    v4 = sp_interp(cog_width_t, cog_flux_t, flux_frac[3]*flux_t_tot, kind=kind)
    ######## calculate the profile concentration
    if v4!=0:
        c_v = v2/v4
    else:
        c_v = 8    
    cog_b = pd.DataFrame({"width": cog_width_b, "flux":cog_flux_b})
    cog_b2 = cog_b[cog_b["width"]<=v2]
    cog_r = pd.DataFrame({"width": cog_width_r, "flux":cog_flux_r})
    cog_r2 = cog_r[cog_r["width"]<=v2]
    cog_t = pd.DataFrame({"width": cog_width_t, "flux":cog_flux_t})
    cog_t2 = cog_t[cog_t["width"]<=v2]
    ####### calculate the profile shape kappa
    cog_t2_width = cog_t2["width"]/v2
    cog_t2_flux = cog_t2["flux"]/(0.85*flux_t_tot)
    cog_t2_width2 = np.append(cog_t2_width, [0, 1])
    cog_t2_flux2 = np.append(cog_t2_flux, [0, 1])
    cog_t_norm = pd.DataFrame({"width_norm": cog_t2_width2, "flux_norm":cog_t2_flux2}).sort_values("width_norm", ascending=True)
    cog_t_norm_width = cog_t_norm["width_norm"]
    cog_t_norm_flux = cog_t_norm["flux_norm"]
    ############# https://numpy.org/doc/stable/reference/generated/numpy.trapz.html, np.trapz(y, x=None, dx=1.0, axis=- 1)
    kappa = np.trapz(cog_t_norm_flux, cog_t_norm_width)-0.5
    ####### calculate the line width asymmetry
    flux_b_v2 = flux_b_tot - (1-flux_frac[1])*0.5*flux_t_tot
    flux_r_v2 = flux_r_tot - (1-flux_frac[1])*0.5*flux_t_tot
    if len(cog_b2)<=2 or max(cog_b2["flux"])<flux_b_v2 or flux_b_v2<0:
        v2_b = 0
        v2_r = v2
    elif len(cog_r2)<=2 or max(cog_r2["flux"])<flux_r_v2 or flux_r_v2<0:
        v2_b = v2
        v2_r = 0
    else:
        v2_b = sp_interp(cog_b2["width"], cog_b2["flux"], flux_b_v2, kind=kind)
        v2_r = sp_interp(cog_r2["width"], cog_r2["flux"], flux_r_v2, kind=kind)
    ########## flux asymmetry
    if (flux_b_tot <=0) or (flux_r_tot <=0):
        a_f = 10
    elif flux_b_tot>flux_r_tot:
        a_f = flux_b_tot/flux_r_tot
    elif flux_b_tot<=flux_r_tot:
        a_f = flux_r_tot/flux_b_tot
    ######### calculate the profile concentration asymmetry
    cog_b3 = cog_b[cog_b["width"]<=v2/2.0]
    cog_r3 = cog_r[cog_r["width"]<=v2/2.0]
    deg_cog = 1
    if len(cog_b3)<=1:
        fit_blue_cog = [0, 0]
        s_b = 0
    else:
        fit_blue_cog = np.polyfit(cog_b3["width"], cog_b3["flux"], deg_cog)
        s_b = fit_blue_cog[0]
    if len(cog_r3)<=1:
        fit_red_cog = [0, 0]
        s_r = 0
    else:
        fit_red_cog = np.polyfit(cog_r3["width"], cog_r3["flux"], deg_cog)
        s_r = fit_red_cog[0]
    if (s_b <=0) or (s_r <=0):
        a_c = 10
    elif s_b>s_r:
        a_c = s_b/s_r
    elif s_b<=s_r:
        a_c = s_r/s_b
    if len(cog_t2)<=1:
        fit_tot_cog = [0, 0]
        s_t = 0
    else:
        fit_tot_cog = np.polyfit(cog_t2["width"], cog_t2["flux"], deg_cog)
        s_t = fit_tot_cog[0]
    ################### calculate the signal-to-noise ratio of the spectrum
    if V_chan is None:
        V_chan = sp_Vchan(dd_range)
    if N_smo is None:
        N_smo=1
    if sigma*V_chan*np.sqrt(len(dd_final)*N_smo)!=0:
        snr = flux_t_tot/(sigma*V_chan*np.sqrt(len(dd_final)*N_smo))
    else:
        snr =np.nan
    results = {
        ########## central velocity
        "V_c": V_c,
        "flux_Vc": flux_Vc,
        ########## final dataframe
        "dd_final": dd_final,
        "V_starti": V_starti,
        "V_endi": V_endi,
        ########## curve of growth for the blue-shift, red-shift and total spectrum
        "cog_width_b": cog_width_b,
        "cog_flux_b": cog_flux_b,
        "cog_width_r": cog_width_r,
        "cog_flux_r": cog_flux_r,
        "cog_width_t": cog_width_t,
        "cog_flux_t": cog_flux_t,
        "cog_width_t_norm": cog_t_norm_width,
        "cog_flux_t_norm": cog_t_norm_flux,
        ########## integrated part at the flat part of CoG
        "flux_b_tot": flux_b_tot,
        "flux_r_tot": flux_r_tot,
        "flux_t_tot": flux_t_tot,
        ########## line widths
        "v1": v1, 
        "v2": v2, 
        "v3": v3, 
        "v4": v4, 
        "v2_b": v2_b,
        "v2_r": v2_r, 
        ########## asymmetry and concentration parameters
        "a_f": a_f,
        "a_c": a_c,
        "c_v": c_v,
        "kappa": kappa, 
        ########## slopes and fitting parameters of linear fitting to the rising part of CoG
        "s_b": s_b, 
        "s_r": s_r, 
        "s_t": s_t, 
        "fit_blue_cog": fit_blue_cog,
        "fit_red_cog": fit_red_cog,
        "fit_tot_cog": fit_tot_cog,
        ########## SNR of the spectrum
        "snr": snr
        }
    return results

###############################################################################################
###############################################################################################
###############################################################################################
################  generate the fake flux density 
def mock_flux(flux, sigma):
    """
    generate mock flux density array by assuming the Gaussian noise

    Parameters
    ----------
    flux : array
        The initial flux density array.
    sigma : float
        The noise level of the spectrum.

    Returns
    -------
    flux_new : array
        The mock flux density array.

    """
    flux_new = flux.copy()
    flux_new = flux_new+np.random.normal(0, sigma, len(flux))
    return flux_new

###############################################################################################
###############################################################################################
###############################################################################################
################  write the file into ascii file
def sp_write(dd0, filename):
    """
    write the spectrum into ascii file

    Parameters
    ----------
    dd0 : dataframe
        The dataframe including velocity, flux density and weight.
    filename : str
        The file we want to write the file.

    Returns
    -------
    None.

    """
    astro_tab = Table.from_pandas(dd0)
    ascii.write(astro_tab, filename, overwrite=True)
    
###############################################################################################
###############################################################################################
###############################################################################################
################  smooth the spectrum
def sp_smooth(dd0, win_len = 11):
    """
    smooth the spectrum using the Hanning window
    Caution: smooth does not change the weight of the spectrum

    Parameters
    ----------
    dd0 : TYPE
        DESCRIPTION.
    win_len : TYPE, optional
        DESCRIPTION. The default is 11.

    Returns
    -------
    None.

    """
    if win_len<3:
        return dd0
    smo_window = np.hanning(win_len)
    flux = dd0["flux"]
    dd_smo = dd0.copy()
    flux_smo = np.convolve(flux, smo_window/smo_window.sum(), mode = "same")
    dd_smo["flux"] = flux_smo
    return dd_smo

###############################################################################################
###############################################################################################
###############################################################################################
#################### calculate the statistical uncertainty of the measurements
def sp_stat(dd0, sigma, V_c0, mc_times=None, sigma_thres=None, maxiters=None, weight_fileter = None, threshold=None, iters=None, V_start=None, V_end = None, ascending = True,\
                  mirror=False, mirror_notes=None, na_values = None, \
                  mask=False, vel_l=None, vel_r=None, set_flux = None, set_weight=None,\
                  deltaV = None, f_ratio = None, m_ratio = None, V_diff = None, fix_range=True, decimal_n = None,\
                  fix_Vc=True, n_extd = None, kind=None, flux_frac = None, N_smo=None):
    """
    calculte the statistical uncertainty by performing Monte Carlo Simulation

    Parameters
    ----------
    dd0 : dataframe
        The final dataframe after running the sp_cog, dd_final. 
        In this way, we could fix the range of signal, thus remove the uncertainties due to signal detection.
    sigma : TYPE
        DESCRIPTION.
    V_c0 : TYPE
        DESCRIPTION.
    fix_Vc : bool, optional
        DESCRIPTION. The default is True.
    iter_times : TYPE, optional
        DESCRIPTION. The default is None.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if mc_times is None:
        mc_times = 50
    if sigma_thres is None:
        sigma_thres = 1.5
    if maxiters is None:
        maxiters = 3
    vel = dd0["velocity"]
    flux = dd0["flux"]
    weight = dd0["weight"]
    Vc_mc = []
    flux_tot_mc = []
    v1_mc = []
    v2_mc = []
    v3_mc = []
    v4_mc = []
    af_mc = []
    ac_mc = []
    as_mc = []
    cv_mc = []
    kappa_mc = []
    NN =  len(flux)
    cog_width_t_mc = np.full((mc_times, NN), np.nan)
    cog_flux_t_mc = np.full((mc_times, NN), np.nan)
    for i in range(mc_times):
        ########## (1) generate mock spectrum
        flux_mc = mock_flux(flux, sigma)
        dd_mc = dd0.copy()
        dd_mc["flux"] = flux_mc
        ######### (2) measure the noise level
        dd_wgt, f_sigma = sp_sigma(dd_mc, V_c0, deltaV = deltaV, weight_fileter = weight_fileter, threshold=threshold, iters=iters, V_start=V_start, V_end = V_end, vel_l=vel_l, vel_r=vel_r)
        ######### (3) drop nan value and order the spectrum
        dd_dropna = sp_drop(dd_mc, weight_fileter = weight_fileter, na_values = na_values)
        dd_order = sp_order(dd_dropna, ascending = ascending)
        ######### (4) mirror the spectrum if needed and mask the spectrum
        if mirror == True:
            dd_mirror, Vc_mirror = sp_mirror(dd_order, V_mirror=V_c0, mirror_notes=mirror_notes, weight_fileter = weight_fileter, na_values = na_values)
            if mask == True:
                dd_mask = sp_mask(dd_mirror, vel_l=vel_l, vel_r=vel_r, set_flux = set_flux, set_weight=set_weight, V_mirror=V_mirror, mirror_notes=mirror_notes)
        elif mask == True:
            dd_mask = sp_mask(dd_order, vel_l=vel_l, vel_r=vel_r, set_flux = set_flux, set_weight=set_weight, V_mirror=V_c0, mirror_notes=mirror_notes)
        else:
            dd_mask = dd_order.copy()
        ######### (5) determine the channels of emission 
        dd_range, dd_signal = sp_range(dd_mask, V_c0, deltaV = deltaV, V_start=V_start, V_end=V_end, f_ratio = f_ratio, m_ratio = m_ratio, V_diff = V_diff, fix_range= fix_range)
        ######### (6) calculate the flux density weighted central velocity
        V_chan = sp_Vchan(dd_range, decimal_n = decimal_n)
        Vc_wgt = sp_Vc(dd_signal, f_sigma)
        ######### (7) build the CoG
        rst_cog = sp_cog(dd_range, dd_signal, V_c0, f_sigma, fix_Vc=fix_Vc, n_extd = n_extd, kind=kind, V_chan=V_chan, flux_frac = flux_frac, N_smo=N_smo) 
        ######### since we fix V_c0, so the uncertainty of V_c is for the flux density weighted central velocity
        Vc_mc.append(Vc_wgt)
        flux_tot_mc.append(rst_cog["flux_t_tot"])
        v1_mc.append(rst_cog["v1"])
        v2_mc.append(rst_cog["v2"])
        v3_mc.append(rst_cog["v3"])
        v4_mc.append(rst_cog["v4"])
        af_mc.append(rst_cog["a_f"])
        ac_mc.append(rst_cog["a_c"])
        cv_mc.append(rst_cog["c_v"])
        kappa_mc.append(rst_cog["kappa"])
        cog_width_t_mc[i, 0:len(rst_cog["cog_width_t"])] = rst_cog["cog_width_t"]
        cog_flux_t_mc[i, 0:len(rst_cog["cog_flux_t"])] = rst_cog["cog_flux_t"]
    ########## use sigma clipping to derive the statistical uncertainty
    Vc_sc = sigma_clip(Vc_mc, sigma=sigma_thres, maxiters=maxiters)
    flux_tot_sc = sigma_clip(flux_tot_mc, sigma=sigma_thres, maxiters=maxiters)
    v1_sc = sigma_clip(v1_mc, sigma=sigma_thres, maxiters=maxiters)
    v2_sc = sigma_clip(v2_mc, sigma=sigma_thres, maxiters=maxiters)
    v3_sc = sigma_clip(v3_mc, sigma=sigma_thres, maxiters=maxiters)
    v4_sc = sigma_clip(v3_mc, sigma=sigma_thres, maxiters=maxiters)
    af_sc = sigma_clip(af_mc, sigma=sigma_thres, maxiters=maxiters)
    ac_sc = sigma_clip(ac_mc, sigma=sigma_thres, maxiters=maxiters)
    cv_sc = sigma_clip(cv_mc, sigma=sigma_thres, maxiters=maxiters)
    kappa_sc = sigma_clip(kappa_mc, sigma=sigma_thres, maxiters=maxiters)
    ########## calculate the standard deviation of the Monce Carlo results as the final statistical results
    Vc_statErr = std(Vc_sc)
    flux_tot_statErr = std(flux_tot_sc)
    v1_statErr = std(v1_sc)
    v2_statErr = std(v2_sc)
    v3_statErr = std(v3_sc)
    v4_statErr = std(v4_sc)
    af_statErr = std(af_sc)
    ac_statErr = std(ac_sc)
    cv_statErr = std(cv_sc)
    kappa_statErr = std(kappa_sc)
    statErr = {
        "Vc_statErr": Vc_statErr,
        "flux_tot_statErr": flux_tot_statErr,
        "v1_statErr": v1_statErr,
        "v2_statErr": v2_statErr,
        "v3_statErr": v3_statErr,
        "v4_statErr": v4_statErr,
        "af_statErr": af_statErr,
        "ac_statErr": ac_statErr,
        "cv_statErr": cv_statErr,
        "kappa_statErr": kappa_statErr,
        "flux_tot_mc": flux_tot_mc,
        "cog_width_t_mc": cog_width_t_mc,
        "cog_flux_t_mc": cog_flux_t_mc,}
    return statErr

# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# #################### build the class -- connect all fucntions together
class spectrum(object):
    """
    build the curve of growth, plot the final results
    """
    ################### initiate the class function
    def __init__(self, dd0):
        """
        read the fits file 

        Parameters
        ----------
        dd0 : dataframe
        The raw dataframe.

        Returns
        -------
        None.

        """
        ########### the raw spectrum
        self.dd0 = dd0
        self.vel0 = dd0["velocity"]
        self.flux0 = dd0["flux"]
        self.weight0 = dd0["weight"]
        
    def class_cog(self, V_c0, weight_fileter = None, threshold=None, iters=None, V_start=None, V_end = None, ascending = True,\
                  mirror=False, mirror_notes=None, V_mirror = None, na_values = None, \
                  mask=False, vel_l=None, vel_r=None, set_flux = None, set_weight=None,\
                  deltaV = None, f_ratio = None, m_ratio = None, V_diff = None, fix_range = False, decimal_n = None,\
                  fix_Vc=True, n_extd = None, kind=None, flux_frac = None, N_smo=None):
        """
        build the curve of growth:
            (1) measure the noise level
            (2) order the spectrum
            (3) mirror the spectrum
            (4) mask the spectrum
            (5) determine the channels of emission
            (6) measure the channel spacing and flux density weighted central velocity
            (7) build the curve of growth

        Parameters
        ----------
        weight_fileter : TYPE, optional
            DESCRIPTION. The default is None.
        threshold : TYPE, optional
            DESCRIPTION. The default is None.
        iters : TYPE, optional
            DESCRIPTION. The default is None.
        V_start : TYPE, optional
            DESCRIPTION. The default is None.
        V_end : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dd_wgt : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.

        """
        ################ (1) measure the noise level
        dd0 = self.dd0
        self.V_c0 = V_c0
        dd_wgt, sigma = sp_sigma(dd0, V_c0, deltaV = deltaV, weight_fileter = weight_fileter, threshold=threshold, iters=iters, V_start=V_start, V_end = V_end, vel_l=vel_l, vel_r=vel_r)
        self.sigma = sigma
        self.dd_wgt = dd_wgt
        ################ (2) drop nan and order the spectrum
        dd_dropna = sp_drop(dd0, weight_fileter = weight_fileter, na_values = na_values)
        dd_order = sp_order(dd_dropna, ascending = ascending)
        self.dd_order = dd_order
        ################ (3) mirror the spectrum if needed and mask the spectrum
        self.vel_l = vel_l
        self.vel_r = vel_r
        if mirror == True:
            self.mirror = True
            dd_mirror, Vc_mirror = sp_mirror(dd_order, V_mirror=V_c0, mirror_notes=mirror_notes, weight_fileter = weight_fileter, na_values = na_values)
            if mask == True:
                self.mask = True
                dd_mask = sp_mask(dd_mirror, vel_l=vel_l, vel_r=vel_r, set_flux = set_flux, set_weight=set_weight, V_mirror=V_mirror, mirror_notes=mirror_notes)
            else:
                self.mask = False
                dd_mask = dd_mirror.copy()
        elif mask == True:
            self.mirror = False
            self.mask = True
            dd_mask = sp_mask(dd_order, vel_l=vel_l, vel_r=vel_r, set_flux = set_flux, set_weight=set_weight, V_mirror=V_c0, mirror_notes=mirror_notes)
        else:
            self.mirror = False
            self.mask = False
            dd_mask = dd_order.copy()
        self.dd_mask = dd_mask
        ################ (4) determine the channels of emission
        dd_range, dd_signal = sp_range(dd_mask, V_c0, deltaV = deltaV, V_start=V_start, V_end=V_end, f_ratio = f_ratio, m_ratio = m_ratio, V_diff = V_diff, fix_range=fix_range)
        self.dd_range = dd_range
        self.dd_signal = dd_signal
        ################ (5) measure the channel spacing and flux density weighted central velocity
        V_chan = sp_Vchan(dd_range, decimal_n = decimal_n)
        Vc_wgt = sp_Vc(dd_signal, sigma)
        self.V_chan = V_chan
        self.Vc_wgt = Vc_wgt
        ################ (6) build the curve of growth
        self.fix_Vc = fix_Vc
        rst_cog = sp_cog(dd_range, dd_signal, V_c0, sigma, fix_Vc=fix_Vc, n_extd = n_extd, kind=kind, V_chan=V_chan, flux_frac = flux_frac, N_smo=N_smo)
        self.rst_cog = rst_cog
        self.V_c = rst_cog["V_c"]
        self.flux_Vc = rst_cog["flux_Vc"]
        ########## final dataframe
        self.dd_final = rst_cog["dd_final"]
        self.V_start = rst_cog["V_starti"]
        self.V_end = rst_cog["V_endi"]
        ########## curve of growth for the blue-shift, red-shift and total spectrum
        self.cog_width_b = rst_cog["cog_width_b"]
        self.cog_flux_b = rst_cog["cog_flux_b"]
        self.cog_width_r = rst_cog["cog_width_r"]
        self.cog_flux_r = rst_cog["cog_flux_r"]
        self.cog_width_t = rst_cog["cog_width_t"]
        self.cog_flux_t = rst_cog["cog_flux_t"]
        self.cog_width_t_norm = rst_cog["cog_width_t_norm"]
        self.cog_flux_t_norm = rst_cog["cog_flux_t_norm"]
        ########## integrated part at the flat part of CoG
        self.flux_b_tot = rst_cog["flux_b_tot"]
        self.flux_r_tot = rst_cog["flux_r_tot"]
        self.flux_t_tot = rst_cog["flux_t_tot"]
        ########## line widths
        self.v1 = rst_cog["v1"]
        self.v2 = rst_cog["v2"]
        self.v3 = rst_cog["v3"]
        self.v4 = rst_cog["v4"]
        self.v2_b = rst_cog["v2_b"]
        self.v2_r = rst_cog["v2_r"]
        ########## asymmetry and concentration parameters
        self.a_f = rst_cog["a_f"]
        self.a_c = rst_cog["a_c"]
        self.c_v = rst_cog["c_v"]
        self.kappa = rst_cog["kappa"]
        ########## slopes and fitting parameters of linear fitting to the rising part of CoG
        self.s_b = rst_cog["s_b"]
        self.s_r = rst_cog["s_r"]
        self.s_t = rst_cog["s_t"]
        self.fit_blue_cog = rst_cog["fit_blue_cog"]
        self.fit_red_cog = rst_cog["fit_red_cog"]
        self.fit_tot_cog = rst_cog["fit_tot_cog"]
        ########## SNR of the spectrum
        self.snr = rst_cog["snr"]
        return rst_cog
        
    def class_stat(self, mc_times=None, sigma_thres=None, maxiters=None, weight_fileter = None, threshold=None, iters=None, V_start=None, V_end = None, ascending = True,\
                  mirror=False, mirror_notes=None, na_values = None, \
                  mask=False, vel_l=None, vel_r=None, set_flux = None, set_weight=None,\
                  deltaV = None, f_ratio = None, m_ratio = None, V_diff = None, fix_range=False, decimal_n = None,\
                  fix_Vc=True, n_extd = None, kind=None, flux_frac = None, N_smo=None):                  
        dd0 = self.dd_final
        sigma = self.sigma
        V_c0 = self.V_c0
        fix_Vc = self.fix_Vc
        statErr = sp_stat(dd0, sigma, V_c0, fix_Vc=fix_Vc, mc_times=mc_times, sigma_thres=sigma_thres, maxiters=maxiters, weight_fileter = weight_fileter,\
                          threshold=threshold, iters=iters, V_start=V_start, V_end = V_end, ascending = ascending, mirror=mirror, mirror_notes=mirror_notes,\
                          na_values = na_values, mask=mask, vel_l=vel_l, vel_r=vel_r, set_flux = set_flux, set_weight=set_weight, deltaV = deltaV, f_ratio = f_ratio,\
                          m_ratio = m_ratio, V_diff = V_diff, fix_range = fix_range, decimal_n = decimal_n, n_extd = n_extd, kind=kind, flux_frac = flux_frac, N_smo=N_smo)
        self.Vc_statErr = statErr["Vc_statErr"]
        self.flux_tot_statErr = statErr["flux_tot_statErr"]
        self.v1_statErr = statErr["v1_statErr"]
        self.v2_statErr = statErr["v2_statErr"]
        self.v3_statErr = statErr["v3_statErr"]
        self.v4_statErr = statErr["v4_statErr"]
        self.af_statErr = statErr["af_statErr"]
        self.ac_statErr = statErr["ac_statErr"]
        self.cv_statErr = statErr["cv_statErr"]
        self.kappa_statErr = statErr["kappa_statErr"]
        self.flux_tot_mc = statErr["flux_tot_mc"]
        self.cog_width_t_mc = statErr["cog_width_t_mc"]
        self.cog_flux_t_mc = statErr["cog_flux_t_mc"]
        return statErr
    
    def cog_plot(self, pdf_file, le_name, czi, w50i, w20i, ax1=None, ax2=None, ax3=None):
        """
        

        Parameters
        ----------
        pdf_file : str
            The name of figure file we want to save.
        le_name : str
            The galaxy name, also legend name.
        czi : float
            The H I central velocity from the ALFALFA catalogue.
        w50i : float
            The line width W50 from the ALFALFA catalogue.
        w20i : float
            The line width W20 from the ALFALFA catalogue.
        ax1 : TYPE, optional
            DESCRIPTION. The default is None.
        ax2 : TYPE, optional
            DESCRIPTION. The default is None.
        ax3 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if (ax1 is None) & (ax2 is None) & (ax3 is None):
            plt.close()
            fig = plt.figure(figsize=[21,5])
            gs1 = gridspec.GridSpec(1,4,width_ratios=[1,1,1, 1])
            gs1.update(left=0.01, right=0.98, top = 0.92, bottom=0.205, hspace=0, wspace=0.25)
            ax1 = plt.subplot(gs1[0,0])
            ax2 = plt.subplot(gs1[0,1])
            ax3 = plt.subplot(gs1[0,2])
            ax4 = plt.subplot(gs1[0,3])
        
        dd0 = self.dd0
        dd0_dpn = dd0.dropna(axis=0).reset_index(drop=True)
        vel_dpn = dd0_dpn.velocity
        flux_dpn = dd0_dpn.flux
        dd_wgt = self.dd_wgt
        dd_final = self.dd_final
        ########## the fixed optical velocity
        V_c0 = self.V_c0
        flux_Vc = self.flux_Vc
        ######### the flux density weighted central velocity
        Vc_wgt = self.Vc_wgt
        ######## czi: the H I heliocentric velocity from ALFALFA
        sigma = self.sigma
        vel_l = self.vel_l
        vel_r = self.vel_r
        vel_wgt = dd_wgt["velocity"]
        flux_wgt = dd_wgt["flux"]
        vel_final = dd_final["velocity"]
        flux_final = dd_final["flux"]
        # ax1_xmin = min(vel_final)
        # ax1_xmax = max(vel_final)
        w_max =  max(w50i, w20i)
        ax1_xmin = min(min(vel_final), czi-1.2*w_max)
        ax1_xmax = max(max(vel_final), czi+1.2*w_max)
        ax1_ymin = -sigma*1.8
        ax1_ymax = max(flux_final)*1.2
        if flux_Vc<-sigma:
            flux_Vc = 3*sigma
        
        ############# plot the raw spectrum, the detected signal and noise level
        ax1.hlines(0, ax1_xmin, ax1_xmax, color="black", linewidth=0.5, zorder=0, label=None)
        ax1.step(vel_dpn, flux_dpn, color="grey",linewidth=3, where="mid", label=None, zorder=1)
        ax1.step(vel_final, flux_final, color="black",linewidth=3, where="mid", label=le_name, zorder=2)
        ax1.vlines(V_c0, -sigma, flux_Vc, color="black", linestyle="--", linewidth=3, zorder=4, label=None)
        ax1.vlines(czi, -sigma, flux_Vc, color="red", linewidth=3, zorder=3, label=None)
        ax1.vlines(Vc_wgt, -sigma, flux_Vc, color="yellow", linewidth=3, zorder=4, label=None)
        ax1.vlines(czi-w50i/2.0,  -sigma, 0.2*ax1_ymax, color="blue", linestyle="-", linewidth=3, zorder=3, label=None)
        ax1.vlines(czi+w50i/2.0,  -sigma, 0.2*ax1_ymax, color="blue", linestyle="-", linewidth=3, zorder=3, label=None)
        ax1.vlines(czi-w20i/2.0,  -sigma, 0.2*ax1_ymax, color="green", linestyle="-", linewidth=3, zorder=3, label=None)
        ax1.vlines(czi+w20i/2.0,  -sigma, 0.2*ax1_ymax, color="green", linestyle="-", linewidth=3, zorder=3, label=None)
        ax1.hlines(0, ax1_xmin, ax1_xmax, color="black", linewidth=0.5, zorder=0, label=None)
        ax1.fill_between(vel_final, sigma, -sigma, color="grey", alpha=0.4, zorder=2, label=r"1 $\sigma$")
        ######### show the masked region
        if vel_l is not None:
            for i in range(len(vel_l)):
                mask_xi = np.linspace(vel_l[i], vel_r[i], 10)
                mask_y1 = ax1_ymin
                mask_y2 = ax1_ymax
                ax1.fill_between(mask_xi, mask_y1, mask_y2, color="lightgrey", hatch = '/')
        ax1.legend(bbox_to_anchor=(1.0, 1.02), loc="upper right", fontsize=16, markerfirst=False, framealpha=0.01)
        ax1.text(0.05, 0.90, '(a)',horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes, fontsize=18)
        ax1.set_xlabel(r"$V\ \mathrm{(km\ s^{-1})}$", fontsize=20)
        ax1.set_ylabel(r"$F_{V}\ \mathrm{(mJy)}$", fontsize=20)
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='major', length=8, width=2., direction="in", labelsize=18)
        ax1.tick_params(axis="both", which="minor", length=4, width=1, direction="in")
        ax1.axis([ax1_xmin, ax1_xmax, ax1_ymin, ax1_ymax])
        
        flux_t_tot = self.flux_t_tot
        cog_width_t = self.cog_width_t
        cog_flux_t = self.cog_flux_t
        V85 = self.v2
        V25 = self.v4
        ax2_xmin = 0
        ax2_xmax = min(V85*2, ax1_xmax-ax1_xmin)
        ax2_ymin = 0.0001
        ax2_ymax = int(min(max(cog_flux_t/flux_t_tot*100), 119))+4.9
        
        ############# plot the curve of growth
        ax2.plot(cog_width_t, cog_flux_t/flux_t_tot*100, color="black", linewidth=3, linestyle="-", zorder=2, label=r"$F_{t}(V)$")
        flux_tot_mc = self.flux_tot_mc
        cog_width_t_mc = self.cog_width_t_mc
        cog_flux_t_mc = self.cog_flux_t_mc
        for i in range(len(flux_tot_mc)):
            ax2.plot(cog_width_t_mc[i, :], cog_flux_t_mc[i, :]/flux_tot_mc[i]*100,color="black", alpha=0.05, linewidth=0.1,linestyle="-", zorder=1, label=None)
        ax2.hlines(100, -5, ax2_xmax,color="black", alpha=1, linewidth=2, linestyle="-.", label=None, zorder=6)
        ax2.hlines(85, -5, V85, color="green", alpha=1, linewidth=3, linestyle="-.", label=r"$V_{\mathrm{85}}$", zorder=7)
        ax2.vlines(V85, 0, 85,color="green", alpha=1, linewidth=3, linestyle="-.", label=None, zorder=7)
        ax2.hlines(25, -5, V25, color="purple", alpha=1, linewidth=3, linestyle="-.", label=r"$V_{\mathrm{25}}$", zorder=7)
        ax2.vlines(V25, 0, 25,color="purple", alpha=1, linewidth=3, linestyle="-.", label=None, zorder=7)
        ax2.text(0.05, 0.90, '(b)',horizontalalignment='left',verticalalignment='top',transform=ax2.transAxes, fontsize=18)
        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='major', length=8, width=2., direction="in", labelsize=18)
        ax2.tick_params(axis="both", which="minor", length=4, width=1, direction="in")
        ax2.set_yticks([20, 40, 60, 80, 100])
        ax2.set_yticklabels(["20", "40", "60", "80", "100"],fontsize=18)
        ax2.set_xlabel(r"$V\ \mathrm{(km\ s^{-1})}$", fontsize=20)
        ax2.set_ylabel(r"$f\ \mathrm{(\%)}$", fontsize=20)
        ax2.legend(fontsize=12, loc='lower right', markerfirst=False, framealpha=0.01)
        ax2.axis([ax2_xmin, ax2_xmax, ax2_ymin, ax2_ymax])
        
        flux_b_tot = self.flux_b_tot
        cog_width_b = self.cog_width_b
        cog_flux_b = self.cog_flux_b
        flux_r_tot = self.flux_r_tot
        cog_width_r = self.cog_width_r
        cog_flux_r = self.cog_flux_r
        fit_blue_cog = self.fit_blue_cog
        fit_red_cog = self.fit_red_cog
        s_b = self.s_b
        s_r =self.s_r
        cog_width_b = np.asarray(cog_width_b)
        cog_width_r = np.asarray(cog_width_r)
        cog_width_b_fit = cog_width_b[cog_width_b<V85]
        cog_width_r_fit = cog_width_r[cog_width_r<V85]
        fit_blue_func = np.poly1d(fit_blue_cog)
        fit_red_func = np.poly1d(fit_red_cog)
        cog_flux_b_fit = fit_blue_func(cog_width_b_fit)
        cog_flux_r_fit = fit_red_func(cog_width_r_fit)
        ax3_xmax = max(V85, cog_width_b[-1], cog_width_r[-1])
        
        a_f = self.a_f
        a_c = self.a_c
        c_v = self.c_v
        kappa = self.kappa
        ############# plot the asymmetry
        ax3.plot(cog_width_b,cog_flux_b/flux_t_tot*100,color="blue", linewidth=3,linestyle="-", zorder=1, label=r"$F_{b}(V)$")
        ax3.plot(cog_width_r,cog_flux_r/flux_t_tot*100,color="red", linewidth=3,linestyle="-", zorder=2, label=r"$F_{r}(V)$")
        ax3.plot(cog_width_b_fit, cog_flux_b_fit/flux_t_tot*100, color="blue", linewidth=2,linestyle="--", zorder=3, label=r"$F_{{b, fit}}(V)$")
        ax3.plot(cog_width_r_fit, cog_flux_r_fit/flux_t_tot*100, color="red", linewidth=2,linestyle="--", zorder=4, label=r"$F_{{r, fit}}(V)$")
        ax3.text(0.05, 0.90, r"(c) $A_F$ = "+ str("%.2f"%a_f),horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes, fontsize=18)
        ax3.text(0.15, 0.82, r"$A_C$ = "+ str("%.2f"%a_c),horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes, fontsize=18)
        ax3.minorticks_on()
        ax3.tick_params(axis='both', which='major', length=8, width=2., direction="in", labelsize=18)
        ax3.tick_params(axis="both", which="minor", length=4, width=1, direction="in")
        ax3.set_yticks([20, 40, 60, 80,100])
        ax3.set_yticklabels(["20", "40", "60", "80", "100"],fontsize=18)
        ax3.set_xlabel(r"$V\ \mathrm{(km\ s^{-1})}$", fontsize=20)
        ax3.set_ylabel(r"$f\ \mathrm{(\%)}$", fontsize=20)
        ax3.legend(fontsize=12, loc='lower right', markerfirst=False, framealpha=0.01)
        ax3.axis([ax2_xmin, ax3_xmax, ax2_ymin, ax2_ymax])
        
        cog_width_t_norm = self.cog_width_t_norm
        cog_flux_t_norm = self.cog_flux_t_norm
        power_x = np.linspace(-0.1, 1.1, 30)
        ax4_xmin, ax4_xmax, ax4_ymin, ax4_ymax = [0, 1.05, 0.0001, 1.05]
        ax4.plot(power_x, power_x, color="black", linewidth=1,linestyle="--", zorder=0, label=None)
        ax4.plot(cog_width_t_norm, cog_flux_t_norm,  color="black", linewidth=3,linestyle="-", zorder=1, label=None)
        ax4.text(0.05, 0.90, r'(d) $C_V$ = '+ str("%.2f"%c_v),horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes, fontsize=18)
        if kappa<0:
            ax4.text(0.15, 0.82, r'$K$ = $-$'+ str("%.3f"%abs(kappa)),horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes, fontsize=18)
        else:
            ax4.text(0.15, 0.82, r'$K$ = '+ str("%.3f"%kappa),horizontalalignment='left',verticalalignment='top',transform=ax4.transAxes, fontsize=18)
        ax4.minorticks_on()
        ax4.tick_params(axis='both', which='major', length=8, width=2., direction="in", labelsize=18)
        ax4.tick_params(axis="both", which="minor", length=4, width=1, direction="in")
        ax4.set_xlabel(r"$V/V_{85}$", fontsize=20)
        ax4.set_ylabel(r"$F_{t}(V)/F_{t}(V_{85})$", fontsize=20)
        ax4.axis([ax4_xmin, ax4_xmax, ax4_ymin, ax4_ymax])
        
        plt.savefig(pdf_file, bbox_inches="tight")
        plt.close()
    

    def rst_write_noErr(self, txt_file, ga_name):
        """
        write the results without running Monte Carlo Simulations

        Parameters
        ----------
        txt_file : TYPE
            DESCRIPTION.
        ga_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        V_c0 = self.V_c0
        Vc_wgt = self.Vc_wgt
        V_c = self.V_c
        flux_t_tot = self.flux_t_tot
        flux_b_tot = self.flux_b_tot
        flux_r_tot = self.flux_r_tot
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        v4 = self.v4
        a_f = self.a_f
        a_c = self.a_c
        c_v = self.c_v
        kappa = self.kappa
        sigma = self.sigma
        snr = self.snr
        mirror_notes = self.mirror
        mask_notes = self.mask
        rst_tab = Table(names= ("ga_name", "V_c0", "Vc_wgt", "V_c", "flux_b_tot", "flux_r_tot", "flux_t_tot", \
                                "v95", "v85", "v75", "v25", \
                                "a_f", "a_c", "c_v", "kappa","sigma", "snr", "mirror_notes", "mask_notes"),\
                        dtype=("S30", "f8", "f8", "f8", "f8", "f8", "f8", \
                                "f8", "f8", "f8", "f8", \
                                "f8", "f8", "f8", "f8", "f8", "f8", "bool", "bool"))
        rst_tab.add_row((ga_name, V_c0, Vc_wgt, V_c, flux_b_tot, flux_r_tot, flux_t_tot, \
                                v1, v2, v3, v4, \
                                a_f, a_c, c_v, kappa, sigma, snr, mirror_notes, mask_notes))
        rst_tab.write(txt_file, format="ascii", overwrite=True)
        
    def rst_write(self, txt_file, ga_name):
        V_c0 = self.V_c0
        Vc_wgt = self.Vc_wgt
        V_c = self.V_c
        Vc_statErr = self.Vc_statErr
        flux_t_tot = self.flux_t_tot
        flux_b_tot = self.flux_b_tot
        flux_r_tot = self.flux_r_tot
        flux_tot_statErr = self.flux_tot_statErr
        v1 = self.v1
        v1_statErr = self.v1_statErr
        v2 = self.v2
        v2_statErr = self.v2_statErr
        v3 = self.v3
        v3_statErr = self.v3_statErr
        v4 = self.v4
        v4_statErr = self.v4_statErr
        a_f = self.a_f
        af_statErr = self.af_statErr
        a_c = self.a_c
        ac_statErr = self.ac_statErr
        c_v = self.c_v
        cv_statErr = self.cv_statErr
        kappa = self.kappa
        kappa_statErr = self.kappa_statErr
        sigma = self.sigma
        snr = self.snr
        mirror_notes = self.mirror
        mask_notes = self.mask
        rst_tab = Table(names= ("ga_name", "V_c0", "Vc_wgt", "V_c", "Vc_statErr", "flux_b_tot", "flux_r_tot", "flux_t_tot", "flux_tot_statErr",\
                                "v95", "v95_statErr", "v85", "v85_statErr", "v75", "v75_statErr", "v25", "v25_statErr", \
                                "a_f", "af_statErr", "a_c", "ac_statErr", "c_v", "cv_statErr", "kappa", "kappa_statErr", \
                                   "sigma", "snr", "mirror_notes", "mask_notes"),\
                        dtype=("S30", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",\
                               "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",\
                                   "f8", "f8", "f8", "f8","f8", "f8", "f8", "f8", \
                                    "f8", "f8", "bool", "bool"))
        rst_tab.add_row((ga_name, V_c0, Vc_wgt, V_c, Vc_statErr, flux_b_tot, flux_r_tot, flux_t_tot, flux_tot_statErr,\
                                v1, v1_statErr, v2, v2_statErr, v3, v3_statErr, v4, v4_statErr, \
                                a_f, af_statErr, a_c, ac_statErr, c_v, cv_statErr, kappa, kappa_statErr, sigma, snr, mirror_notes, mask_notes))
        rst_tab.write(txt_file, format="ascii", overwrite=True)
        
        
    
   
        
    
    


