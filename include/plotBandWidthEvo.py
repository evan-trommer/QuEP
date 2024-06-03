# This script takes the bottom and top electrons from a probing band and 

import os
import numpy as np
import matplotlib.colors as col
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import rc
import pdb
import math
import copy
import time
import progressbar
import multiprocessing as mp
import include.simulations.useQuasi3D as sim

plt.rcParams.update({'font.size': 12})
#plt.rcParams['animation.ffmpeg_path'] = '/ffmpeg/bin'
plt.rcParams['figure.constrained_layout.use'] = True
mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

t0 = sim.getTime()
# Definition of Constants
M_E = 9.109e-31                      # Electron rest mass in kg
EC = 1.60217662e-19                  # Electron charge in C
EP_0 = 8.854187817e-12               # Vacuum permittivity in C/(V m)
C = 299892458                        # Speed of light in vacuum in m/s

W_P = sim.getPlasFreq()
plasma_bnds = sim.getBoundCond()
propspeed = sim.getPropagationSpeed()

def returnXi(z):
    return z - t0*propspeed

def returnZ(xi):
    return xi + t0*propspeed

def get_momenta(xi_f,x_f,y_f,z_f,px_f,py_f,pz_f, w, topM, bottomM):
     
    #With (px,py,pz) = (110,0,0), and plasma density 1e15, the outline of the e- probe will have a maximum radius at xi =-7.7   
    # old value was 44.3
    xi1 = z_f[np.round(z_f, 2) == 45] - t0
    y1 = y_f[np.round(z_f, 2) == 45]
    py1 = py_f[np.round(z_f, 2) == 45]
    px1 = px_f[np.round(z_f, 2) == 45]
    num = len(xi1)
    top_indicies = []
    bottom_indicies = []
    
    #f = open('bandIndicie.txt', 'r+')
    #f.truncate(0)
    #f.close()
    
    #txt_file = open('bandIndicie.txt', 'a')
    for i in range(num):
        if np.round(y1, 2)[i] == topM: 
            #txt_file.write(f"Y-coord for top of band: {y1[i]}, px : {px1[i]}, py : {py1[i]}, index: {i} \n") 
            top_indicies.append(i)
        if np.round(y1, 2)[i] == bottomM: 
            #txt_file.write(f"Y-coord for bottom of band: {y1[i]}, px : {px1[i]}, py : {py1[i]}, index: {i} \n")
            bottom_indicies.append(i)

    #txt_file.close()
    
    #top_indicies = np.array(top_indicies)
    #bottom_indicies = np.array(bottom_indicies)
    
    i_top = top_indicies[0]
    i_bot = bottom_indicies[0]
    
    pytop, pztop, pxtop = py_f[i_top], pz_f[i_top], px_f[i_top]
    pybot, pzbot, pxbot = py_f[i_bot], pz_f[i_bot], px_f[i_bot]
    
    return num, pytop, pztop, pxtop, pybot, pzbot, pxbot

def initializeScreens():
    
    xstart_mm = 0
    xend_mm = 100
    xstep_mm = 1
    
    xstart = xstart_mm*W_P*(10**(-3))/C
    xend = (xend_mm)*W_P*(10**(-3))/C
    xstep = xstep_mm*W_P*(10**(-3))/C
    
    xScreens = np.arange(xstart, xend, xstep)
    xScreens_mm = np.arange(xstart_mm, xend_mm+1, xstep_mm)
    
    #print(len(xScreens),len(xScreens_mm))
    
    return xScreens, xScreens_mm

#def createData(x_f,y_f,z_f,px_f,py_f,pz_f):
    
#    return x_f1, y_f1, z_f1, px_f, py_f, pz_f
    

#def findAvgPy(py):
#    avg_py = py
#    return avg_py

def plotWidths(x_screens, x_screensmm, py_top, pz_top, px_top, py_bot, pz_bot, px_bot, topM, bottomM, dr, N):
    
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=300)
    ax.set_xlabel('$x$ (mm)', size = 16) 
    ax.set_ylabel(r'$\sigma_y$ ($c/\omega_p$)', size = 16)
    fig.suptitle(r"Band-Width vs. $x$ for $\delta y = 0.1 c/\omega_p$", size = 18)
    
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_xticks([0,25,50, 75, 100])
    ax.set_yticks([0,0.05,0.1])

    #ax.xaxis.set_label_coords(0.54,-0.02)
    #ax.yaxis.set_label_coords(-0.02,0.5)
    
    y_top = findAvgY.yTraj(topM, px_top, -abs(py_top), x_screens, x_screens[0])
    y_bot = findAvgY.yTraj(bottomM, px_bot, py_bot, x_screens, x_screens[0])
    
    ywidth = y_top-y_bot 
    #print(y_top)
    #print(y_bot)
    #print(ywidth)
    
    colors = ['red', 'orange', 'green', 'c', 'navy', 'blue', 'm', 'violet']
    
    ax.errorbar(x_screensmm, np.abs(ywidth), xerr=1, yerr=0.002, color = colors[N], fmt = '-o', label = "Simulated Width")
    #ax.axhline(y = 0, xmin = 0, xmax = 1, color = 'k', ls = '-.', lw = 0.75, label = r"$\sigma_y = 0$")
    fig.legend(loc='lower right', bbox_to_anchor=(1.0, 0.3), prop={'size': 6})
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filenumber = "{:05.1f}".format(bottomM + (dr*N)).replace(".","-")
    fig.savefig(f'band-widths-y0{filenumber}.png',dpi=300,transparent=False)
    print("Plot of band widths saved!")
    
    ax.cla()
    fig.clf()
    plt.close(fig)