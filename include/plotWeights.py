# Script for generating 2D plots of electron trajectories
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
mpl.use('Agg')
plt.rcParams.update({'font.size': 10 })
plt.rcParams['figure.constrained_layout.use'] = True

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Definition of Constants
M_E = 9.109e-31                      # Electron rest mass in kg
EC = 1.60217662e-19                  # Electron charge in C
EP_0 = 8.854187817e-12               # Vacuum permittivity in C/(V m)
C = 299892458                        # Speed of light in vacuum in m/s

WB = False # Sequential
Viridis = True # Sequential + Perceptually Uniform
BuPu = False # Sequential
Jet = False

t0 = sim.getTime()

propspeed = sim.getPropagationSpeed()

def returnXi(z):
    return z - t0*propspeed

def returnZ(xi):
    return xi + t0*propspeed




def plotcross(w_export1, x_0, y_0, xi_0, z_0, s1, s2, ydensity, xidensity, beamxi_c, sigma_xi):
# Plot w (w_x) vs xi
    ##########################################################################################
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    ax3.plot(xi_0[int(xidensity*39):int(xidensity*40)-1],w_export1,"o", label="weighting_function",alpha=0.7)
    
    #ax3.legend(loc='upper right')
    ax3.set_xlabel("$\\xi_0$ ($c/\omega_p$)")
    ax3.set_ylabel("$w$")
    ax3.set_title("Combined weighting")

    print(f"y_export = {y_0[xidensity*39]} , {y_0[xidensity*40-1]}")

    Deltaxi = 2*s2/xidensity #same as xstep
    summ = 0
    for w_xiy in w_export1:
        summ += w_xiy*Deltaxi
    print(f"Summ = {summ}")
    #ax3.text(-13,0,f"Sum $w$ * $\Delta \\xi$ = {summ:.3f}", fontdict=None, horizontalalignment='center', fontsize=10)

    # Plot the gaussian line for expected
    xi = np.linspace(-18,-8,1000)
    Ymod = np.exp((-1.*(0.21-0)**2)/(2*0.2**2))
    w_exp = Ymod * np.exp((-1.*(xi-beamxi_c)**2)/(2*sigma_xi**2))
    ax3.plot(xi,w_exp, label="Expected")

    fig3.legend(loc=2, prop={'size': 8})
    plt.tight_layout()

    fig3.savefig('Gaussian-weights_xi-cross-direction-1.png',dpi=600,transparent=False)

def ploty(w_y, x_0, y_0, xi_0, z_0, s1, s2, ydensity, xidensity, beamy_c, sigma_y):
# Plot w_y vs y
    ##########################################################################################
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(y_0[0:len(y_0):xidensity],w_y,"o", label="Weighting function",alpha=0.7)
    
    #ax3.legend(loc='upper right')
    ax4.set_xlabel("$y_0$ ($c/\omega_p$)")
    ax4.set_ylabel("$w_y$")
    ax4.set_title("y-direction weighting")

    Deltay = 2*s1/ydensity
    summ = 0
    for w_y_i in w_y:
        summ += w_y_i*Deltay
    #ax4.text(0,0.2,f"Sum $w_y$ * $\Delta y$ = {summ:.3f}", fontdict=None, horizontalalignment='center', fontsize=10)

    # Plot the gaussian line for expected
    y = np.linspace(-1,1,1000)
    w_xi_exp = np.exp((-1.*(y-beamy_c)**2)/(2*sigma_y**2))
    ax4.plot(y,w_xi_exp, label="Expected")

    fig4.legend(loc=2, prop={'size': 8})

    plt.tight_layout()

    fig4.savefig('Gaussian-weights_y-direction-1.png',dpi=600,transparent=False)


def plotxi(w_xi, x_0, y_0, xi_0, z_0, s1, s2, ydensity, xidensity, beamxi_c, sigma_xi):
# Plot w (w_xi) vs xi
    ##########################################################################################
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    
    # Plot the data from sim
    ax5.plot(xi_0[0:len(w_xi)],w_xi,"o", label="Weighting function",alpha=0.7)
    
    #ax5.legend(loc='upper right')
    ax5.set_xlabel("$\\xi_0$ ($c/\omega_p$)")
    ax5.set_ylabel("$w_\\xi$")
    ax5.set_title("$\\xi$-direction weighting")

    Deltaxi = 2*s2/xidensity
    summ = 0
    for w_xi_i in w_xi:
        summ += w_xi_i*Deltaxi
    #ax5.text(-13,0.2,f"Sum $w_\\xi$ * $\Delta \\xi$ = {summ:.3f}", fontdict=None, horizontalalignment='center', fontsize=10)

    # Plot the gaussian line for expected
    xi = np.linspace(-18,-8,1000)
    w_xi_exp = np.exp((-1.*(xi-beamxi_c)**2)/(2*sigma_xi**2))
    ax5.plot(xi,w_xi_exp, label="Expected")

    fig5.legend(loc=2, prop={'size': 8})
    plt.tight_layout()

    fig5.savefig('Gaussian-weights_xi-direction-1.png',dpi=600,transparent=False)




def plotweightsxiy(y_0,xi_0, w, rand):
    
    path = os.getcwd()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_path = os.path.join(path,f'animation-{timestr}-{rand}')
    os.mkdir(new_path)

    ximin = -40 #36  #25#27#400
    ximax = -10 #50  #500
    
    ymin = -1
    ymax = 1

    bin_resolution = .1 #0.02 #c/w_p
    bin_edges_xi = np.arange(ximin, ximax, 100)
    bin_edges_y = np.arange(ymin, ymax, 100)
    
    cmin = 1       # Minimum density displayed
    vmin_ = cmin    # Minimum color value
    vmax_ = 100    # Maximum color value

    if (WB):
        cmap = plt.cm.binary
    elif (Viridis):
        cmap = plt.cm.viridis
    elif (BuPu):
        cmap = plt.cm.BuPu
    elif (Jet):
        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_under(color='white')
    else:
        cmap = plt.cm.gist_gray
    norm = mpl.colors.Normalize(vmin=1, vmax=400)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(8, 5), dpi=600)
    fig.suptitle("Weighting Map")
    #plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    h = ax.hist2d(xi_0[:], y_0[:], weights=w[:], bins=[100,100])#, bins=(bin_edges_xi,bin_edges_y), cmap=cmap, vmin=vmin_,vmax=vmax_,cmin=cmin)#, norm=norm)

    ax.set_ylim(-1,1)
    #ax.set_xlim(ximin,ximax)

    if (WB):
        ax.set_facecolor('white')
    #elif (Viridis):
    #    ax.set_facecolor('#30013b')
    else:
        ax.set_facecolor('white')

    ax.set(xlabel = '$\\xi$ ($c/\omega_p$)', ylabel = 'Y ($c/\omega_p$)')

    secax = ax.secondary_xaxis('top', functions= (returnZ, returnXi))
    secax.set(xlabel= 'Z ($c/\omega_p$)')
    
    cbar = plt.colorbar(h[3], ax=ax, orientation='horizontal')
    #cbar.set_label('Electron Density')

    #Saving
    filename = str(os.path.join(new_path,f'weighting-xi-y.png'))
    fig.savefig(filename,dpi=600,transparent=False)
        
    ax.cla()
    fig.clf()
    plt.close(fig)
