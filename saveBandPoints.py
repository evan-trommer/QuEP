# Script for calculating particle trajectories
import os
import numpy as np
import pdb
import math
import copy
import time
import progressbar
import multiprocessing as mp
import matplotlib.colors as col
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import rc
import include.simulations.useQuasi3D as sim
import pandas as pd

plt.rcParams.update({'font.size': 12 })
plt.rcParams['figure.constrained_layout.use'] = True
mpl.use('Agg')

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

def Gamma(p):
    return np.sqrt(1.0 + p**2)

def Velocity(px,ptot):
# Returns relativistic velocity from momentum
    return px / Gamma(ptot)

def getBallisticTraj(x_0,y_0,z_0,px,py,pz,x_s):
# Use ballistic matrix to find positions on screens
    dx = x_s - x_0
    y_f = y_0 + dx * (py/px)
    z_f = z_0 + dx * (pz/px)

# Find time traveled to get proper xi
    p = np.sqrt(px**2 + py**2 + pz**2)
    vx = Velocity(px, p)
    vy = Velocity(py, p)
    vz = Velocity(pz, p)
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    dtot = np.sqrt((x_s - x_0)**2 + (y_f - y_0)**2 + (z_f - z_0)**2)
    t = dtot/vtot

    #xi_f = xi_0 + dx * (pz/px) + t

    return y_f, z_f

def getYTraj(x_0,y_0,px,py,x_s):
# Use ballistic matrix to find positions on screens
    dx = x_s - x_0
    y_f = y_0 + dx * (py/px)

    return y_f

def focalLength(y0): 
    
    p2 = 110**2 
    gamma = 110 
    k = 0.475
    rb = 0.65
    conversion = 0.01/0.06
    f0 = conversion*p2/(2*gamma*k*rb)
    focL = f0/np.sqrt(1-(np.divide(y0,rb)**2))
    
    return focL

def focalLength2(y0, px, py, pz): 
    
    p2 = px**2 + py**2 + pz**2
    gamma = np.sqrt(p2)
    k = 0.475
    rb = 0.65
    conversion = 0.01/0.06
    f0 = conversion*p2/(2*gamma*k*rb)
    focL = f0/np.sqrt(1-(np.divide(y0,rb)**2))
    
    return focL

def trueFocalLength(y0, px, py): 
    
    conversion = 0.01/0.06
    focL = -conversion*np.multiply(y0, np.divide(px, py))

    return focL

def sigY(x, y, dy): 
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    f0 = conversion*p2/(2*gamma*k*rb)
    focBottom = f0/np.sqrt(1-(np.divide(y,rb)**2))
    focTop = f0/np.sqrt(1-(np.divide(y+dy,rb)**2))
    numerator = y*(focTop-focBottom)-(dy*focBottom)
    denominator = dy*np.multiply(focBottom,focTop)
    
    yWidth = dy*np.abs(1 + x*np.divide(numerator, denominator))
    
    return yWidth

def sigYSmallBandApprox(x, y, dy): 
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    f0 = conversion*p2/(2*gamma*k*rb)
    y0normSquared = np.square(np.divide(y, rb))
    
    yWidth = dy*np.abs(1 + ((y0normSquared/np.sqrt(1-y0normSquared)) - np.sqrt(1-y0normSquared))*x/f0)
    
    return yWidth

def sigYFullApprox(x, y, dy): 
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    f0 = conversion*p2/(2*gamma*k*rb)
    foc = f0/np.sqrt(1-(np.divide(y,rb)**2))
    
    yWidth = dy*np.abs(1-np.divide(x,foc))
    
    return yWidth

def minBandWidthDistance(y0, dy):
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    f0 = conversion*p2/(2*gamma*k*rb)
    focBottom = f0/np.sqrt(1-(np.divide(y0,rb)**2))
    focTop= f0/np.sqrt(1-(np.divide(y0+dy,rb)**2))
    denominator = focBottom - np.multiply(y0, (np.divide(np.subtract(focTop,focBottom), dy)))
    
    xmin = np.divide(np.multiply(focBottom, focTop), denominator)
    
    return xmin

def minBandWidthDistanceThinBandApprox(y0, dy):
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    f0 = conversion*p2/(2*gamma*k*rb)
    y0normSquared = np.square(np.divide(y0, rb))
    
    xmin = f0*np.divide(np.sqrt(1-y0normSquared), 1-2*y0normSquared)
    
    return xmin

def minBandWidthDistanceFullApprox(y0, dy):
    
    p2 = 110**2
    gamma = 110
    k = 0.475 
    rb = 0.65
    conversion = 0.01/0.06 #convert c/w_p to mm
    foc = conversion*(p2/(2*gamma*k*rb))/np.sqrt(1-(np.divide(y0,rb)**2))
    
    xmin = np.multiply(foc,(1+np.divide(np.square(y0), (rb**2 - np.square(y0)))))
    
    return xmin

def prepare(sim_name,shape_name,noObj,rand, y0):
# Plot evolution of probe after leaving plasma
    if (sim_name.upper() == 'OSIRIS_CYLINSYMM'):
        import include.simulations.useOsiCylin as sim
    elif (sim_name.upper() == 'QUASI3D'):
        import include.simulations.useQuasi3D as sim
    else:
        print("Simulation name unrecognized. Quitting...")
        exit()

    W_P = sim.getPlasFreq()
    plasma_bnds = sim.getBoundCond()
    shape_name = shape_name.capitalize()

# Plot slices
# For bin size = 0.006 (lambda/10)
# Run 130 Limits: (27,52), (-6,6), Bins: (4167,2000)
#         (35,40), (-1,1), Bins: (833,333)
# For bin size = 0.03
# Run 130 Limits: (27,52), (-6,6), Bins: (833,400)
# Run 232 Limits: (435,475), (0,6), Bins: (1333,200)

    # Choose boundaries of screens in mm
    xstart_mm = 0
    xend_mm = 100
    xstep_mm = 1

    #binsizez = 6500//4#6000#833#2833#4167#1000#2666#1333
    #binsizey = 1000//4#400#2000#160#666#200
    
    # For Quasi_ID = 000130, use (36,48) for front blowout regime, or (28,50) for whole wakefield structure
    ximin = -14
    ximax = -4.5
    
    zmin =  ximin+t0#28  #25#27#400
    zmax =  ximax+t0#50  #500
    
    ymin = -1.0
    ymax = 1.0

    bin_resolution = 0.01 #0.02 #c/w_p
    bin_edges_z = np.arange(zmin, zmax, bin_resolution)
    bin_edges_y = np.arange(ymin, ymax, bin_resolution)
    
    ######## END PARAMETERS ########

    # Normalize screen distances
    screen_dists = list(range(xstart_mm,xend_mm+1,xstep_mm))#np.arange(xstart_mm,xend_mm+1,xstep_mm)#
    #print(f"First screen location in mm is x = {screen_dists[0]}, and last screen is at x = {screen_dists[-1]}")
    slices = len(screen_dists) # Number of Screens
    xs_norm = []
    for i in range(0,slices):
        xs_norm.append(screen_dists[i] * W_P * 10**(-3) / C)
    #print(f"First screen located at x = {xs_norm[0]} and last screen recorded at x = {xs_norm[-1]}")

    # Generate arrays of coordinates at origin + each screen
    yslice = np.empty([noObj])
    zslice = np.empty([noObj])

    # Get cwd and create path variable for frame output
    path = os.getcwd()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_path = os.path.join(path,f'probe-prop-y0{y0:.02f}-dy0p048'.replace('.','p'))
    folderExist = os.path.exists(new_path)
    if folderExist != True:
        os.mkdir(new_path)

    return plasma_bnds, slices, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists

def getBandData(i,x_f,y_f,z_f,px_f,py_f,pz_f, w, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM):
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        assert (abs(xs_norm[i]) == plasma_bnds[2]), "Want to start at screen x=0 in order to save the correct data."
    else: 
        yslice = y_f
        zslice = z_f
        print("Starting at screen x = 0mm. Saving band points ... ")
        
   
    wNonZero = np.array((np.nonzero(w))[0])
    tempX = np.array(x_f[wNonZero])
    tempY = np.array(yslice[wNonZero])
    tempZ = np.array(zslice[wNonZero])
    w = np.array(w)
    tempW = np.array(w[wNonZero])
    tempPy = np.array(py_f[wNonZero])
    tempPz = np.array(pz_f[wNonZero])
    tempPx = np.array(px_f[wNonZero])
    
    tempInd = np.where((tempY <= topM+0.001) & (tempY >= bottomM-0.001))[0]
    tempX = np.array(tempX[tempInd])
    tempY = np.array(tempY[tempInd])
    tempZ = np.array(tempZ[tempInd])
    tempW = np.array(tempW[tempInd])
    tempPy = np.array(tempPy[tempInd])
    tempPz = np.array(tempPz[tempInd])
    tempPx = np.array(tempPx[tempInd])
    
    df = pd.DataFrame({"Non-zero weight indicies": tempInd, "Y points": tempY, "Z-points": tempZ})#({"Y points" : ySliver, "N_e" : Ny})
    df.to_csv("nonzero_points.csv", sep=':', index=False)
    
    wNonZero = None
    
    return tempX, tempY, tempZ, tempW, tempPx, tempPy, tempPz

def individualParticleExtremumData(i,bandX,bandY,bandZ,bandPx,bandPy,bandPz, y0_mask, new_path,singleData, screen_dists, topM, bottomM):

    if (i != 0): 
        assert (i == 0), "Cannot save initial points at x != 0."

    z_center = 44.39 
    slice_width = 0.025 # create band of length dz = 0.10 c/w_p
    z_low = z_center - slice_width
    z_high = z_center + slice_width
    ySliver = np.array(bandY[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    zSliver = np.array(bandZ[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    pxSliver = np.array(bandPx[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    pySliver = np.array(bandPy[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    pzSliver = np.array(bandPz[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    
    bandIndicies = np.argsort(ySliver)
    highPointsY = np.array(ySliver[bandIndicies][-1])
    lowPointsY = np.array(ySliver[bandIndicies][0])
    highPointsZ = np.array(zSliver[bandIndicies][-1])
    lowPointsZ = np.array(zSliver[bandIndicies][0])
    
    pxTop = np.array(pxSliver[np.where(ySliver == highPointsY)[0]][0])
    pyTop = np.array(pySliver[np.where(ySliver == highPointsY)[0]][0])
    pzTop = np.array(pzSliver[np.where(ySliver == highPointsY)[0]][0])
    pxBottom = np.array(pxSliver[np.where(ySliver == lowPointsY)[0]][0])
    pyBottom = np.array(pySliver[np.where(ySliver == lowPointsY)[0]][0])
    pzBottom = np.array(pzSliver[np.where(ySliver == lowPointsY)[0]][0])
    
    df = pd.DataFrame({'Band height' :[y0_mask], 'y top': highPointsY,'px Top':pxTop , 'py Top':pyTop, 'pz Top':pzTop, 'y bot': lowPointsY,'px Bottom':pxBottom , 'py Bottom':pyBottom, 'pz Bottom':pzBottom })
    df.to_csv(singleData, sep=',', mode = 'a', index=False, header=False)
    
    return highPointsY, highPointsZ, pxTop, pyTop, pzTop, lowPointsY,lowPointsZ, pxBottom, pyBottom, pzBottom

def getCalibrationPoints(i,bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM):
    
    #fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300)
    #fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    fig2, ax2_2 = plt.subplots(1, figsize=(8, 5), dpi=600)
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        yslice, zslice = getBallisticTraj(bandX, bandY, bandZ, bandPx,bandPy, bandPz, xs_norm[i])
    else:
        yslice = bandY
        zslice = bandZ
    
    zcenters = np.linspace(44, 45, 26)
    ytop = []
    ztop = []
    ybot = []
    zbot = []
    Deltaz = []
    Deltay = []
    delz = 0.002 #is an ideal value
    ytop2 = []
    ybot2 = []
    

    for j in range(len(zcenters)):
        
        z_c = zcenters[j]
        z_low = z_c-delz
        z_high = z_c+delz
        zSliver = np.array(zslice[np.where((z_low <= zslice) & (zslice <= z_high))])
        ySliver = np.array(yslice[np.where((z_low <= zslice) & (zslice <= z_high))])
        
        bandIndicies = np.argsort(ySliver)
        highPointsY = np.array(ySliver[bandIndicies][-1])
        lowPointsY = np.array(ySliver[bandIndicies][0])
        highPointsZ = np.array(zSliver[bandIndicies][-1])
        lowPointsZ = np.array(zSliver[bandIndicies][0])
        
        ytop.append(highPointsY)
        ztop.append(highPointsZ)
        ybot.append(lowPointsY)
        zbot.append(lowPointsZ)
        
        ytop2.append(np.max(highPointsY))
        ybot2.append(np.max(highPointsY))
        
        sigY = np.abs(highPointsY-lowPointsY)
        sigZ = np.abs(highPointsZ-lowPointsZ)
        Deltaz.append(sigZ)
        Deltay.append(sigY)
    
        #print(f"sigY = {sigY}")
        # avgZ = np.average([highPointsZ, lowPointsZ])
        # #print(f"<z> = {avgZ}")
        # delZ = np.abs(highPointsZ-lowPointsZ)
        #print(f"Dz = {delZ}")

        # ax1.plot(zcenters, Deltaz, '-or', markersize = 0.8, label = r'Z-Shift with $\delta z = {:.03f}$'.format(2*delz))
        # #ax1.set_ylim(0.0,0.03)
        # ax1.set_xlim(43.5,45.5)
        # ax1.set_xlabel(r'$z_c$ ($c/\omega_p$)', fontsize = 14)
        # ax1.set_ylabel(r'$\Delta z$ ($c/\omega_p$)', fontsize = 14)
        # fig1.suptitle('Effect of Band ``Wing" on Finding Extremum Values', size = 14)
        # ax1.legend(prop = { "size": 10 }, loc = 'upper right')
        # ax1.tick_params(axis="x", labelsize=12)
        # ax1.tick_params(axis="y", labelsize=12)

        #fig1.savefig(f'.png',dpi=300,transparent=False)
        
    # ax2_1.plot(zcenters, Deltay, '-sr', markersize = 0.8,label = r"Measured")
    # ax2_1.axhline(0.048, 0, 1, color = 'k', ls = '--', label = r'$\delta y = 0.048$')
    # ax2_1.set_ylim(-0.05,0.1)
    # ax2_1.set_xlim(43.5,46.5)
    # ax2_1.set_xlabel(r'$z$ ($c/\omega_p$)', fontsize = 14)
    # ax2_1.set_ylabel(r'$\sigma_y$ ($c/\omega_p$)', fontsize = 14)
    # ax2_1.legend(prop = { "size": 10 }, loc = 'upper right')
    # ax2_1.tick_params(axis="x", labelsize=12)
    # ax2_1.tick_params(axis="y", labelsize=12)
    
    lowestPointY = np.min(ybot)
    lowestPointZ = zcenters[np.where(ybot == np.min(ybot))][0]
    
    ax2_2.plot(zcenters, ytop, '-or', markersize = 0.8,label = "Measured Top")
    ax2_2.plot(zcenters, ybot, '-o', c = 'orange',markersize = 0.8, label = "Measured Bottom")
    ax2_2.plot(lowestPointZ, lowestPointY, 'o', c = 'darkgreen',markersize = 1.5, label = "Lowest Point")
    ax2_2.fill_between(zcenters, ytop, ybot, color = 'r', alpha = 0.6)
    ax2_2.axhline(0.00, 0, 1, color = 'k', ls = '--', label = r"$y_0 = 0.00$")
    ax2_2.axhline(0.048, 0, 1, color = 'k', ls = '--', label = r"$y_0+\delta y = 0.048$" )
    ax2_2.axvline(44.5, 0, 1, color = 'magenta', ls = '--', label = r"$z_c = 44.5$" )
    ax2_2.axvline(44.4, 0, 1, color = 'indigo', ls = '--', label = r"$z_c = 44.4$" )
    ax2_2.axvline(44.3, 0, 1, color = 'navy', ls = '--', label = r"$z_c = 44.3$" )
    ax2_2.set_ylim(-0.25,0.25)
    ax2_2.set_xlim(43.5,45.5)
    ax2_2.set_xlabel(r'$z$ ($c/\omega_p$)', fontsize = 14)
    ax2_2.set_ylabel(r'$y$ ($c/\omega_p$)', fontsize = 14)
    ax2_2.legend(prop = { "size": 10 }, loc = 'upper right')
    ax2_2.tick_params(axis="x", labelsize=12)
    ax2_2.tick_params(axis="y", labelsize=12)

        #fig1.savefig(f'offCenterPoints_overZ_X-{screen_dists[i]}mm.png',dpi=300,transparent=False)
        #fig.savefig(f'bandSliver-atX-1mm_zc-{z_center:.01f}.png',dpi=600,transparent=False)
    fig2.suptitle(r'Measured Band Width vs $Z$', size = 14)
    filename = str(os.path.join(new_path,f'bandWidth_overZ_X-0{screen_dists[i]}mm.png'))
    fig2.savefig(f'{filename}',dpi=600,transparent=False)
        

def plotBands(i,bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM, ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, ybot_0,zbot_0, pxbot_0, pybot_0, pzbot_0): 
    
    fig, ax = plt.subplots(1, figsize=(8, 5), dpi=600)
    fig.suptitle("Progression of Band Probe")
        
    # Project positions at distances in x_s
    # If x_s out of plasma, use ballistic trajectory
    #print(xs_norm[0], xs_norm[-1])
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        yslice, zslice = getBallisticTraj(bandX, bandY, bandZ, bandPx,bandPy, bandPz, xs_norm[i])
        ytop, ztop = getBallisticTraj(bandX[0], ytop_0, ztop_0, pxtop_0,pytop_0, pztop_0, xs_norm[i])
        ybot, zbot = getBallisticTraj(bandX[0], ybot_0, zbot_0, pxbot_0,pybot_0, pzbot_0, xs_norm[i])
    else:
        yslice = bandY
        zslice = bandZ
        ytop, ztop = ytop_0, ztop_0
        ybot, zbot = ybot_0, zbot_0
    
    bin_resolution = 0.02 #0.02 #c/w_p
    bin_edges_z = np.arange(zmin, zmax, bin_resolution)
    bin_edges_y = np.arange(ymin, ymax, bin_resolution)
    
    cmin = 1       # Minimum density displayed
    vmin_ = cmin    # Minimum color value
    vmax_ = 100#250
    cmap = plt.cm.plasma
    
    zmin = 38
    zmax = 48
    ymin = -1.0
    ymax = 1.0
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    h = ax.hist2d(zslice[:], yslice[:], weights=bandW[:], bins=(bin_edges_z,bin_edges_y), cmap=cmap, vmin=vmin_,vmax=vmax_,cmin=cmin)#, norm=norm)
    temptext = ax.text(zmin+2,ymax*0.75,f"x = {screen_dists[i]:03}mm", fontdict=None, horizontalalignment='left', fontsize=16, color="Black")
    ax.plot(zbot, ybot, 'o', color = 'r', ls = None, markersize = 1.8)
    ax.plot(ztop, ytop, 'o', color = 'b', ls = None, markersize = 1.8)
    
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(zmin,zmax)
    if (WB):
        ax.set_facecolor('white')
    #elif (Viridis):
    #    ax.set_facecolor('#30013b')
    else:
        ax.set_facecolor('white')

    #ax.set(xlabel = 'Z ($c/\omega_p$)', ylabel = 'Y ($c/\omega_p$)')
    fig.suptitle(fr'Band Probe Propagation', size = 16)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    
    ax.set_xlabel(r'Z ($c/\omega_p$)', size = 14)
    ax.set_ylabel(r'Y ($c/\omega_p$)', size = 14)
    
    cbar = plt.colorbar(h[3], ax=ax, orientation='horizontal')#, pad=0.2)
    #cbar.set_label('Electron Density')

    #Saving
    filenumber = "{:05.1f}".format(screen_dists[i]).replace(".","-")
    filename = str(os.path.join(new_path,f'progression-x-{filenumber}mm.png'))
    fig.savefig(f'bandProgression-x-{filenumber}mm.png',dpi=600,transparent=False)
        
    ax.cla()
    fig.clf()
    plt.close(fig)
    
def propagateBandlet(i, bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, plasma_bnds, xs_norm, yslice, zslice, screen_dists, topM, bottomM, dataFile, momentumFile, new_path, include_plots):
    
    #fig, ax = plt.subplots(1,figsize=(8, 5), dpi=600)#figsize=(6, 4), dpi=300)
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        yslice, zslice = getBallisticTraj(bandX, bandY, bandZ, bandPx, bandPy, bandPz, xs_norm[i])
    else:
        yslice = bandY
        zslice = bandZ

    # z_c1 = np.linspace(44, 46, 21)  #points along Z to center the bandlet
    # for k in range(len(z_c1)):
    #     print(f"z_center = {z_c1[k]}")
        
    z_center = 44.39#44.28#44.39
    slice_width = 0.025 # create band of length dz = 0.10 c/w_p
    z_low = z_center - slice_width
    z_high = z_center + slice_width
    zSliver = np.array(zslice[np.where((z_low <= zslice) & (zslice <= z_high))])
    ySliver = np.array(yslice[np.where((z_low <= zslice) & (zslice <= z_high))])
    pxSliver = np.array(bandPx[np.where((z_low <= zslice) & (zslice <= z_high))])
    pySliver = np.array(bandPy[np.where((z_low <= zslice) & (zslice <= z_high))])
    
    bandIndicies = np.argsort(ySliver)
    highPointsY = np.array(ySliver[bandIndicies][-1])
    lowPointsY = np.array(ySliver[bandIndicies][0])
    tempYMax = highPointsY
    highPointsZ = np.array(zSliver[bandIndicies][-1])
    lowPointsZ = np.array(zSliver[bandIndicies][0])
    if (highPointsY >= 0 and lowPointsY < 0): 
        highPointsY = lowPointsY
        lowPointsY = tempYMax
        
    pxTop = np.array(pxSliver[np.where(highPointsY == ySliver[bandIndicies])[0]][0])
    pyTop = np.array(pySliver[np.where(highPointsY == ySliver[bandIndicies])[0]][0])
    pxBottom = np.array(pxSliver[np.where(lowPointsY == ySliver[bandIndicies])[0]][0])
    pyBottom = np.array(pySliver[np.where(lowPointsY == ySliver[bandIndicies])[0]][0])
    
    sigY = np.abs(highPointsY-lowPointsY)
        
    # avgZ = np.average([highPointsZ, lowPointsZ])
    # #print(f"<z> = {avgZ}")
    # delZ = np.abs(highPointsZ-lowPointsZ)
    #print(f"Dz = {delZ}")

    df = pd.DataFrame({'Screen in mm' : [screen_dists[i]], 'ytop' : highPointsY, 'ybot' : lowPointsY, "Y-width": sigY})
    df.to_csv(dataFile, sep=',', mode = 'a', index=False, header=False)
    
    df2 = pd.DataFrame({'Screen in mm' : [screen_dists[i]], 'px Top':pxTop , 'py Top':pyTop, 'px Bottom':pxBottom , 'py Bottom':pyBottom })
    df2.to_csv(momentumFile, sep=',', mode = 'a', index=False, header=False)
    
    #print("Data saved! Plotting data . . .")
    
    
    # ax.plot(highPointsZ, highPointsY,  'ob', markersize = 2.5, label = 'Highest point of band')
    # ax.plot(lowPointsZ, lowPointsY,  'or', markersize = 2.5, label = 'Lowest point of band')
    # ax.axvline(avgZ, 0, 1, color = 'grey', linestyle = '--', linewidth = 0.8, label = r'$z_c = {}$'.format(z_center))
    # temptext = ax.text(43.1,0.56,rf"$\sigma_y$ = {sigY:.4} $c/\omega_p$", fontdict=None, horizontalalignment='left', fontsize=14, color="Black")
    # temptext2 = ax.text(43.1,0.48,rf"$\langle z \rangle$ = {avgZ:.4f} $c/\omega_p$", fontdict=None, horizontalalignment='left', fontsize=14, color="Black")
    # temptext3 = ax.text(43.1,0.40,rf"$\Delta z$ = {delZ:.4} $c/\omega_p$", fontdict=None, horizontalalignment='left', fontsize=14, color="Black")
    
    # ax.set_ylim(-0.65,0.65)
    # ax.set_xlim(43,47)
    
    # ax.set_xlabel(r'Z ($c/\omega_p$)', fontsize = 14)
    # ax.set_ylabel(r'Y ($c/\omega_p$)', fontsize = 14)
    # fig.suptitle('Small Sliver of the Band', size = 14)
    # ax.legend(prop = { "size": 10 }, loc = 'upper right')

    # ax.tick_params(axis="x", labelsize=12)
    # ax.tick_params(axis="y", labelsize=12)

    # #Saving
    # #filenumber = "{:05.1f}".format(screen_dists[i]).replace(".","")
    # filename = str(os.path.join(new_path,f'bandSliver_X-{screen_dists[i]:05.1f}mm_zc-{z_center:.03f}'.replace(".","")))
    # fig.savefig(f'{filename}.png',dpi=600,transparent=False)
    # #fig.savefig(f'bandSliver-atX-1mm_zc-{z_center:.01f}.png',dpi=600,transparent=False)
    # ax.cla()
    # #fig.clf()
    # plt.close(fig)

def justDoIt(i, bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, plasma_bnds, xs_norm, yslice, zslice, screen_dists, topM, bottomM, dataFile, new_path):
    
    fig1, ax1 = plt.subplots(1,figsize=(8, 5), dpi=600)
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        yslice, zslice = getBallisticTraj(bandX, bandY, bandZ, bandPx, bandPy, bandPz, xs_norm[i])
        #upperBand, lowerBand = getYTraj(xExtr[0], yExtr[0],pxExtr[0],pyExtr[0], xs_norm[i]), getYTraj(xExtr[1], yExtr[1],pxExtr[1],pyExtr[1], xs_norm[i])
        #print("New Slice")
    else:
        #print("No Change in y-slice.")
        yslice = bandY
        zslice = bandZ
        #upperBand, lowerBand = bandT, bandB
        
     # z_c1 = np.linspace(44, 46, 21)  #points along Z to center the bandlet
    # for k in range(len(z_c1)):
    #     print(f"z_center = {z_c1[k]}")
        
    z_center = 44.28#44.39
    slice_width = 0.025 # create band of length dz = 0.05 c/w_p
    z_low = z_center - slice_width
    z_high = z_center + slice_width
    zSliver = np.array(zslice[np.where((z_low <= zslice) & (zslice <= z_high))])
    ySliver = np.array(yslice[np.where((z_low <= zslice) & (zslice <= z_high))])
    
    bandIndicies = np.argsort(ySliver)
    highPointsY = np.array(ySliver[bandIndicies][-1])
    lowPointsY = np.array(ySliver[bandIndicies][0])
    highPointsZ = np.array(zSliver[bandIndicies][-1])
    lowPointsZ = np.array(zSliver[bandIndicies][0])
    
    #highPointsY = np.max(ySliver[bandIndicies])
    #lowPointsY = np.min(ySliver[bandIndicies])
    
    minYIndex = np.array(np.where(highPointsY == np.min(highPointsY))[0])
    
    
    sigY = np.abs(highPointsY-lowPointsY)
        
    # avgZ = np.average([highPointsZ, lowPointsZ])
    # #print(f"<z> = {avgZ}")
    # delZ = np.abs(highPointsZ-lowPointsZ)
    #print(f"Dz = {delZ}")
    
    #print("Data saved! Plotting data . . .")
    
    
    # yvals_short, Ny = np.unique(np.sort(ySliver)[::-1], return_counts=True)
    # #print(f"There are {len(Ny)} electrons in the sliver of the band at x={screen_dists[i]}.")
    
    # ax2.plot(yvals_short, Ny, '-k')
    # ax2.text(0.45, 1.9, rf"Total $N_e$ = {np.sum(Ny)}", fontdict=None, horizontalalignment='left', fontsize=12, color="Black")
    # ax2.text(0.45, 1.82, rf"$x$ = {i}mm", fontdict=None, horizontalalignment='left', fontsize=12, color="Black")
    # #ax.text(lowerBand-0.02, 1.8, r"$y_{min}$", fontdict=None, horizontalalignment='left', fontsize=10, color="Black")
    # #ax.text(upperBand+0.05, 1.8, r"$y_{max}$", fontdict=None, horizontalalignment='left', fontsize=10, color="Black")
    # ax2.set_xlabel(r'$y$ ($c/\omega_p$)', size = 12)
    # ax2.set_ylabel(r'$N_e(y)$ ($c/\omega_p$)', size = 12)
    # ax2.set_title(r'$N_e(y)$ for a Bandlet')
    # #ax.axvline(upperBand, 0, 1, c = 'r', ls = '--')
    # #ax.axvline(lowerBand, 0, 1, c = 'b', ls = '--')
    # ax2.set_xlim(-0.5, 0.5)
    # ax2.set_ylim(0,2)
    # ax2.invert_xaxis()
    # #ax[1].scatter(zExtr, yExtr)
    # #ax[1].set_ylabel(r'$y$ ($c/\omega_p$)', size = 12)
    # #ax[1].set_xlabel(r'$z$ ($c/\omega_p$)', size = 12)
    
    # fname = "{:05.1f}".format(screen_dists[i]).replace(".","")
    # fig2.savefig(f'Ne_vsY_x-{fname}mm.png',dpi=600,transparent=False)
        
    #ax.cla()
    #fig.clf()
    #plt.close(fig)
    
def saveSinglePointTrajectory(i, bandX, plasma_bnds, xs_norm, new_path, screen_dists, y0_mask, ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, ybot_0, zbot_0, pxbot_0, pybot_0, pzbot_0, dataFile): 
    
    if (abs(xs_norm[i]) > plasma_bnds[2]):
        ytop, ztop = getBallisticTraj(bandX[0], ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, xs_norm[i])
        ybot, zbot = getBallisticTraj(bandX[0], ybot_0, zbot_0, pxbot_0, pybot_0, pzbot_0, xs_norm[i])
    else:
        #print("No Change in y-slice.")
        ytop, ztop = ytop_0, ztop_0
        ybot, zbot = ybot_0, ztop_0
        
    df = pd.DataFrame({'Screen in mm' : [screen_dists[i]], 'ytop' : ytop, 'ybot' : ybot})
    df.to_csv(dataFile, sep=',', mode = 'a', index=False, header=False)
    
def saveSingleFocusing(new_path, y0_mask, pxtop_0, pytop_0, pztop_0, pxbot_0, pybot_0, pzbot_0, dataFile, dataFile2): 
    
    fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    
    focusData = pd.read_csv(f'{dataFile}', sep=',', header = None, index_col=None, engine = "python")
    xscreen = np.array(focusData[0])
    sortInd = np.argsort(xscreen)
    xscreen = np.array(xscreen[sortInd])
    ytop = focusData[1]
    ytop = np.array(ytop[sortInd])
    ybot = focusData[2]
    ybot = np.array(ybot[sortInd])
    
    focTop_analytic = focalLength2(ytop[0], pxtop_0, pytop_0, pztop_0)
    focBot_analytic = focalLength2(ybot[0], pxbot_0, pybot_0, pzbot_0)
    
    Ytop = ytop[0]-np.multiply(xscreen, np.divide(ytop[0],focTop_analytic))
    Ybot = ybot[0]-np.multiply(xscreen, np.divide(ybot[0],focBot_analytic))
    
    ax1.plot(xscreen, ybot, color = 'darkviolet', marker = 'o', markersize = 2, ls = None, label = r'Measured $Y_{bottom}$')
    ax1.plot(xscreen, ytop, color = 'orangered', marker = 'o',markersize = 2, ls = None, label = r'Measured $Y_{top}$')
    ax1.plot(xscreen, Ybot, color = 'b', ls = '--', label = r'Eq.[3]')
    ax1.plot(xscreen, Ytop, color = 'k', ls = '--',label = r'Eq.[4]')
    ax1.set(xlabel = r'$x$ [mm]', ylabel = r'$y$ ($c/\omega_p$)')
    ax1.set_xlim(-0.1, 101)
    ax1.set_ylim(-0.75, 0.5)
    ax1.legend( fontsize="9")
    fig1.suptitle(r'X-Y Trajectory of Top and Bottom of Band')
    pngfilename = str(os.path.join(new_path,f'xyTrajectory_y0-{y0_mask:.03f}'.replace('.', 'p')))
    fig1.savefig(f'{pngfilename}.png',dpi=300,transparent=False)
    
    if (np.any(ytop <= 0)): 
        f_top = np.min(xscreen[np.where(ytop <= 0)[0]])
    else: 
        f_top = 101
        
    if (np.any(ybot <= 0)): 
        f_bottom = np.min(xscreen[np.where(ybot <= 0)[0]])
    else: 
        f_bottom = 101
    
    df = pd.DataFrame({'y0': y0_mask,'ytop' : [ytop[0]], 'focTop': f_top, 'px Top':pxtop_0, 'py Top':pytop_0, 'pz Top':pztop_0, 'ybot' : [ybot[0]], 'focBot': f_bottom, 'px Bottom':pxbot_0 , 'py Bottom':pybot_0, 'pz Bottom':pzbot_0})
    df.to_csv(f'{dataFile2}', sep=',', mode = 'a', index=False, header=False)

def calculateDerivatives(bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, plasma_bnds, xs_norm, yslice, zslice, screen_dists, topM, bottomM, new_path): 
    
    fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    fig2, ax2 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    fig3, ax3 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300) 

    # if (abs(xs_norm[i]) > plasma_bnds[2]):
    #     yslice, zslice = getBallisticTraj(bandX, bandY, bandZ, bandPx, bandPy, bandPz, xs_norm[i])
    # else:
    #     yslice = bandY
    #     zslice = bandZ
        
    z_center = 44.39#44.28#44.39
    slice_width = 0.025 # create band of length dz = 0.10 c/w_p
    z_low = z_center - slice_width
    z_high = z_center + slice_width
    xSliver = np.array(bandX[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    zSliver = np.array(bandZ[np.where((z_low <= bandZ) & (bandZ <= z_high))])
    ySliver = np.array(bandY[np.where((z_low <=  bandZ) & ( bandZ <= z_high))])
    pxSliver = np.array(bandPx[np.where((z_low <=  bandZ) & (bandZ <= z_high))])
    pySliver = np.array(bandPy[np.where((z_low <=  bandZ) & (bandZ <= z_high))])
    pzSliver = np.array(bandPz[np.where((z_low <=  bandZ) & (bandZ <= z_high))])
    
    bandIndicies = np.argsort(ySliver)
    xSliver = np.array(xSliver[bandIndicies])
    ySliver = np.array(ySliver[bandIndicies])
    zSliver = np.array(zSliver[bandIndicies])
    pxSliver = np.array(pxSliver[bandIndicies])
    pySliver = np.array(pySliver[bandIndicies])
    pzSliver = np.array(pzSliver[bandIndicies])
    
    focus_y = []
    
    for i in range(0,len(ySliver)):
        for j in range(len(screen_dists)):
            if (abs(xs_norm[j]) > plasma_bnds[2]):
                ys, zs = getBallisticTraj(xSliver[i], ySliver[i], zSliver[i], pxSliver[i], pySliver[i], pzSliver[i], xs_norm[j])
            else:
                ys = ySliver[i]
                zs = zSliver[i]
                
            if ys <= 0: 
                focus_y.append(screen_dists[j])
                break
                
    focus_y = np.array(focus_y)
    first_deriv = np.gradient(focus_y, ySliver)
    second_deriv = np.gradient(first_deriv, ySliver)
    
    print(f"y_0 = {ySliver[0]}, f(y_0) = {focus_y[0]}, f'(y_0) = {first_deriv[0]}")
    
    ax1.errorbar(ySliver, focus_y, yerr = 1, fmt='-.b', label = r'$f_{sim}$')
    ax1.set(xlabel = r'$y$ [$c/\omega_p$]', ylabel = r'$f(y_0)$ $[mm]$')
    fig1.suptitle(r"Experimental Focal Lengths Between $y_0$ and $y_0+\delta y$")
    ax1.set_ylim(30, 40)
    ax1.legend()
    fig1.savefig("measured_focals_y-0p36.png", dpi=300)
    
    ax2.plot(ySliver, first_deriv, color = 'r', marker = 'o', markersize = 1.5)
    ax2.set(xlabel = r'$y$ [$c/\omega_p$]', ylabel = r'$\frac{df}{dy_0}$ $[mm/ c/\omega_p]$')
    ax2.set_ylim(-0.01, 0.1)
    fig2.suptitle(r"Experimental First Derivative Between $y_0$ and $y_0+\delta y$")
    fig2.savefig("measured_firstDeriv_y-0p36.png", dpi=300)
    
    ax3.plot(ySliver, second_deriv, color = 'magenta',  marker = 'o', markersize = 1.5)
    ax3.set(xlabel = r'$y$ [$c/\omega_p$]', ylabel = r'$\frac{d^2f}{dy_0^2}$ $[mm/ (c/\omega_p)^2]$')
    fig3.suptitle(r"Experimental Second Derivative Between $y_0$ and $y_0+\delta y$")
    fig3.savefig("measured_secondDeriv_y-0p36.png", dpi=300)


def plotWidthEvo(path, xminfile, txtFiles, pFiles, y0, plotFocals, focalLengthsFile): 
    
    fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    fig2, ax2 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    fig3, ax3 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    fig4, ax4 = plt.subplots(1,figsize=(6, 4), dpi=300)#figsize=(6, 4), dpi=300)
    
    height = y0
    dy = 0.048
    
    #print(fileIndex, files, "# of Sample Points = ", nEdgePoints[fileIndex])
    widthData = pd.read_csv(f'{path}/{txtFiles}', sep=',', header = None, index_col=None, engine = "python")
    
    momentumData = pd.read_csv(f'{path}/{pFiles}', sep=',', header = None, index_col=None, engine = "python")
    
    screenDists = widthData[0]
    bandWidths = widthData[3]
    sortedIndicies = np.argsort(screenDists)
    screenDists = np.array(screenDists[sortedIndicies])
    bandWidths = np.array(bandWidths[sortedIndicies])
    topBandTraj = widthData[1]
    bottomBandTraj = widthData[2]
    topBandTraj = np.array(topBandTraj[sortedIndicies])
    bottomBandTraj = np.array(bottomBandTraj[sortedIndicies])
    yt_0 = topBandTraj[0]
    yb_0 = bottomBandTraj[0]
    
    xs = momentumData[0]
    sortedIndicies2 = np.argsort(xs)
    xs = np.array(xs[sortedIndicies2])
    pxTop = momentumData[1]
    pyTop = momentumData[2]
    pxBot = momentumData[3]
    pyBot = momentumData[4]
    pxTop = np.array(pxTop[sortedIndicies2])
    pyTop = np.array(pyTop[sortedIndicies2])
    pxBot = np.array(pxBot[sortedIndicies2])
    pyBot = np.array(pyBot[sortedIndicies2])

    #measured initial width
    dy_exp = bandWidths[0]
    
    #predicted widths according to formulas made by me
    prediction1 = sigY(screenDists, height, dy)
    prediction1_exp = sigY(screenDists, height, dy_exp)
    prediction2 = sigYSmallBandApprox(screenDists, height, dy)
    prediction2_exp = sigYSmallBandApprox(screenDists, height, dy_exp)
    
    #locating minimum width locations
    minIndex = np.where(bandWidths == np.min(bandWidths))[0]
    xmin = screenDists[minIndex]
    yWidthAtXmin = bandWidths[minIndex]
    #print(f"xmin={xmin}, sigY_min={yWidthAtXmin}")
    df = pd.DataFrame({'Band height':y0 ,'xmin' : xmin,"experimental initial width": dy_exp, "Y-width": yWidthAtXmin})  
    df.to_csv(xminfile, sep=',', mode = 'a', index=False, header=False)
    print("Minimum width position recorded to data file!")
    
    #predicted trajectories according to marisa's ballistic trajectories
    focTop = np.array(focalLength(yt_0))
    focBottom = np.array(focalLength(yb_0))
    #Ytop = -x*(y0+dy)/f(y0+dy) +(y0+dy)
    #Ybot = -x*y0/f(y0) +y0
    ytrajTop_analytic = -np.multiply(np.divide(yt_0,focTop), screenDists) + (height+dy)
    ytrajBottom_analytic = -np.multiply(np.divide(yb_0,focBottom), screenDists) + height
    
    #locating focal lengths 
    if (np.any(bottomBandTraj <= 0)):
        f_y0 = np.max(screenDists[np.where(bottomBandTraj >= 0)[0]])
        print(f_y0)
    elif (height < 0.3): 
        f_y0 = 28
    else: 
        f_y0 = 100
    
    if (np.any(topBandTraj <= 0)):
        f_y0pDy = np.max(screenDists[np.where(topBandTraj >= 0)[0]])
        print(f_y0pDy)
    else: 
        f_y0pDy = 101

    df2 = pd.DataFrame({'Band height':y0 ,'focTop': [f_y0pDy], 'focBot':[f_y0], 'initial top y': [yt_0], 'initial bottom y': [yb_0]})  
    df2.to_csv(focalLengthsFile, sep=',', mode = 'a', index=False, header=False)
    print("Experimental focal lengths recorded to data file!")
    
    if (height <= 0.54): 
        
        ax1.plot(screenDists, prediction1, '-k', lw = 1.2, label = 'Theoretical')
        ax1.plot(screenDists, prediction2, '-g', lw = 1.2, label = 'Theoretical (Small Band Approx.)')
        # ax1.plot(screenDists, prediction1_exp, '-m', lw = 1.2, label = 'Theoretical (exp)')
        # ax1.plot(screenDists, prediction2_exp, '-y', lw = 1.2, label = 'Theoretical (exp) (Small Band Approx.)')
    
    ax1.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured")
    #ax1.plot(xmin, yWidthAtXmin, 'ob', ms = 2, label = 'Point of Min. Width')
    #temptext = ax1.text(30,0.05,r"$y_top = {}$".format(dy_exp), fontdict=None, horizontalalignment='left', fontsize=14, color="Black")
    ax1.set_xlabel(r'$x$ (mm)')
    ax1.set_ylabel(r'$\sigma_y$ ($c/\omega_p$)')
    ax1.set_title(r"Band Width Evolution $\sigma_y(x, y_0, \delta y)$ at $y_0=${0:.3f}".format(height))
    
    #plt.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured")

    ax1.legend()
    pngFile = os.path.join(path,f'BW_evolve.png')
    fig1.savefig(f"{pngFile}", dpi=300)
    ax1.cla()
    
    if (height <= 0.35):
        prediction3 = sigYFullApprox(screenDists, height, dy)
        prediction3_exp = sigYFullApprox(screenDists, height, dy_exp)
        
        ax2.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured")
        ax2.plot(screenDists, prediction3, '-k', lw = 1.2, label = 'Theoretical (Ideal Approx.)')
        # ax2.plot(screenDists, prediction3_exp, '-g', lw = 1.2, label = 'Theoretical (Full Approx.)')
        ax2.plot(xmin, yWidthAtXmin, 'ob', ms = 2, label = 'Point of Min. Width')
        ax2.set_xlabel(r'$x$ (mm)')
        ax2.set_ylabel(r'$\sigma_y$ ($c/\omega_p$)')
        ax2.set_title(r"Band Width Evolution $\sigma_y(x, y_0, \delta y)$ at $y_0=${0:.3f}".format(height))
        
        #plt.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured")

        ax2.legend()
        pngFile2 = os.path.join(path,f'BW_evolve_wApprox.png')
        fig2.savefig(f"{pngFile2}", dpi=300)
        ax2.cla()
    
    if (plotFocals):
        
        ax3.plot(screenDists, topBandTraj, color = 'darkblue', marker = 'o', ms = 1.2, ls = None, label = r"Measured $Y_{top}$")
        ax3.plot(screenDists, bottomBandTraj, color = 'maroon', marker = 'o', ms = 1.2, ls = None,label = r"Measured $Y_{bot}$")
        ax3.plot(screenDists, ytrajTop_analytic, color = 'darkviolet', ls = '--', lw = 0.9, label = r"Analytic $Y_{top}$")
        ax3.plot(screenDists, ytrajBottom_analytic, color = 'orangered', ls = '--', lw = 0.9, label = r"Analytic $Y_{bot}$")
        ax3.set_xlabel(r'$x$ (mm)')
        ax3.set_ylabel(r'$y$ ($c/\omega_p$)')
        ax3.set_title(r"Trajectories of Electrons in Band")
        
        #plt.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured")

        ax3.legend( fontsize="8")
        pngFile3 = os.path.join(path,f'xy-trajectory.png')
        fig3.savefig(f"{pngFile3}", dpi=300)
        ax3.cla()
    
    # ax4.plot(xs, pxTop, color = 'darkblue', marker = 'o', ms = 1.2, ls = None, label = r"Measured $p_{x,t}$")
    # ax4.plot(xs, pxBot, color = 'maroon', marker = 'o', ms = 1.2, ls = None, label = r"Measured $p_{x,b}$")
    # ax4.plot(xs, pyTop, color = 'darkviolet', marker = 'o', ms = 1.2, ls = None, label = r"Measured $p_{y,t}$")
    # ax4.plot(xs, pyBot, color = 'orangered', marker = 'o', ms = 1.2, ls = None, label = r"Measured $p_{y,b}$")
    # ax4.set_xlabel(r'$x$ (mm)')
    # ax4.set_ylabel(r'$p_x$ or $p_y$ ($m_ec$)')
    # ax4.set_title(r"Momenta of Electrons in Band")
    # #plt.plot(screenDists, bandWidths, '-or', ms = 1.2, label = "Measured"
    # ax4.legend( fontsize="8")
    # pngFile4 = os.path.join(path,f'momentum-vsX.png')
    # fig4.savefig(f"{pngFile4}", dpi=300)
    # ax4.cla()
    
        
def plotMinBandWidths(dataF, fFile, plotFocals):
    
    dy = 0.048
    fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    fig2, ax2 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    fig3, ax3 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    fig4, ax4 = plt.subplots(1,figsize=(6, 4), dpi=300)
    fig5, ax5 = plt.subplots(1,figsize=(6, 4), dpi=300)
    fig6, ax6 = plt.subplots(1,figsize=(6, 4), dpi=300)
    fig7, ax7 = plt.subplots(1,figsize=(6, 4), dpi=600)
    fig8, ax8 = plt.subplots(1,figsize=(6, 4), dpi=300)
    fig9, ax9 = plt.subplots(1,figsize=(6, 4), dpi=300)
    

    minWidthData = pd.read_csv(f'{dataF}', sep=',', header = None, index_col=None, engine = "python")
    #print("Data file", minWidthData)
    y0 = np.array(minWidthData[0])
    xmin_data = np.array(minWidthData[1])
    initWidths = np.array(minWidthData[2])
    #minWidths = np.array(minWidthData[3])
    
    y0_05rb = np.array(y0[y0 <= 0.34])
    xmin_data_05rb = np.array(xmin_data[y0 <= 0.34])
    y0_09rb = np.array(y0[y0 <= 0.54])
    # xmin_data_09rb = np.array(xmin_data[y0 <= 0.54])
    y0_1rb = np.array(y0[y0 <= 0.64])
    # xmin_data_1rb = np.array(xmin_data[y0 <= 0.64])
    # y0_135rb = np.array(y0[y0 >= 0.66])
    # xmin_data_135rb = np.array(xmin_data[y0 >= 0.66])
    dy_exp = np.array(initWidths[y0 <= 0.54])
    
    xmin_prediction1 = np.array(minBandWidthDistance(y0_1rb, dy)) #full formula over 0 < y/rb < 1 
    xmin_prediction2 = np.array(minBandWidthDistanceThinBandApprox(y0_1rb, dy)) #thin band approx over 0 < y/rb < 1
    xmin_prediction2_5 = np.array(minBandWidthDistance(y0_09rb, dy))#full formula over 0 < y/rb < 0.9
    # xmin_prediction2_5_2 = np.array(minBandWidthDistanceThinBandApprox(y0_09rb, dy))#thin band approx over 0 < y/rb < 0.9
    # xmin_prediction3_5_3 = np.array(minBandWidthDistanceFullApprox(y0_09rb, dy))#ideal approx over 0 < y/rb < 0.9
    xmin_prediction3 = np.array(minBandWidthDistanceFullApprox(y0_05rb, dy))#ideal approx over 0 < y/rb < 0.5
    
    xmin_pred_exp1 = np.array(minBandWidthDistance(y0_09rb, dy_exp))#full formula over 0 < y/rb < 0.9
    xmin_pred_exp2 = np.array(minBandWidthDistanceThinBandApprox(y0_09rb, dy_exp))#thin band approx over 0 < y/rb < 0.9
    xmin_pred_exp3 = np.array(minBandWidthDistanceFullApprox(y0_09rb, dy_exp))#ideal approx over 0 < y/rb < 0.9
    
    #xMinTot = xmin_data
    
    #minWidth_prediction = sigYApprox(xmin_data, y0, dy)
    
    ax1.errorbar(y0_05rb, xmin_data_05rb, yerr=1, color = 'g', fmt = '-o', ms = 2.5, alpha = 0.75, label = r"Data")
    ax1.plot(y0_1rb, xmin_prediction1, '--k', linewidth=1, label = 'Theoretical') 
    ax1.plot(y0_1rb, xmin_prediction2, '--b', linewidth=1, label = 'Theoretical (Small Band Appx.)') 
    ax1.plot(y0_05rb, xmin_prediction3, '--r', linewidth=1, label = 'Theoretical (Full Appx.)') 
    ax1.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\tilde{f}$ (mm)')
    ax1.set_xlim(-0.02, 0.35)
    ax1.set_ylim(0, 80)
    ax1_x2 = ax1.twiny()
    # set limits for shared axis
    ax1_x2.set_xlim(-0.02/0.65,0.35/0.65)
    ax1_x2.set_xlabel(r"$y_0/r_b$")
    ax1_x2.xaxis.set_label_coords(0.5, 1.05)
    fig1.suptitle(r"Screen Distance of Minimum Band Width $\tilde{f}(y_0, \delta y)$") 
  
    # ax1.legend( fontsize="8", loc="upper left")
    # fig1.savefig(f"minWidthDists_halfRb.png", dpi=300)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
    # ax1.cla()
    # fig1.clf()
    # plt.close(fig1)
    
    # ax7.errorbar(y0_05rb/0.65, xmin_data_05rb, yerr=1, color = 'g', fmt = '-o', linewidth = 2.5,ms = 4.5, alpha = 1, label = r"Data")
    # ax7.plot(y0_09rb/0.65, xmin_pred_exp1, '--k', linewidth=1.75, label = 'Theoretical') 
    # ax7.plot(y0_09rb/0.65, xmin_pred_exp2, '--b', linewidth=1.75, label = 'Thin Band Appx.') 
    # ax7.plot(y0_09rb/0.65, xmin_pred_exp3, '--r', linewidth=1.75, label = 'Thin Band Near Axis Appx.') 
    # ax7.set_xlabel(r'$y_0/r_b$', fontsize = 18)
    # ax7.xaxis.set_label_coords(0.475, 0)
    # ax7.set_ylabel(r'$x_{min}$ (mm)', fontsize = 18)
    # #ax7.set_ylabel(r'$\tilde{f}$ (mm)', fontsize = 18 )
    # ax7.yaxis.set_label_coords(0.0, 0.425)
    # ax7.set_xlim(-0.02, 0.55)
    # ax7.set_ylim(20, 80)
    # ax7.set_xticks([0, 0.075, 0.15, 0.35,0.425,0.5]) 
    # ax7.set_yticks([20, 30, 60, 70, 80]) 
    # ax7.tick_params(axis='both', labelsize=14)
    # # ax7_x2 = ax7.twiny()
    # # set limits for shared axis
    # # ax7_x2.set_xlim(-0.02/0.65,0.35/0.65)
    # # ax7_x2.set_xlabel(r"$y_0/r_b$")
    # # ax7_x2.xaxis.set_label_coords(0.5, 1.05)
    # fig7.suptitle(r"Screen Distances of Minimum Band Width", fontsize = 20, va = 'top') 
  
    # ax7.legend( fontsize="14", loc="upper left")
    # fig7.savefig(f"minWidthDists_halfRb-experimental.png", dpi=600)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
    # ax7.cla()
    # fig7.clf()
    # plt.close(fig7)
    
    # ax2.errorbar(y0_1rb, xmin_data_1rb, yerr=1, color = 'g', fmt = '-o', ms = 2.5, alpha = 0.75, label = "Data")
    # ax2.plot(y0_09rb, xmin_prediction2_5, '--k', linewidth=1, label = 'Theoretical') 
    # ax2.plot(y0_09rb, xmin_prediction3_5_3, '--r', linewidth=1, label = 'Theoretical (Full Approx.)') 
    # #ax1.plot(y0_1rb, xmin_prediction1, '--k', linewidth=1, label = 'Theoretical') 
    # #ax1.plot(y0_1rb, xmin_prediction2, '--r', linewidth=1, label = 'Theoretical (Small Band Appx.)')  
    # ax2.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\tilde{f}$ (mm)')
    # ax2.set_xlim(0.33, 0.65)
    # ax2.set_ylim(-10, 120)
    # ax2_x2 = ax2.twiny()
    # # set limits for shared axis
    # ax2_x2.set_xlim(0.33/0.65,0.65/0.65)
    # ax2_x2.set_xlabel(r"$y_0/r_b$")
    # ax2_x2.xaxis.set_label_coords(0.5, 1.05)
    # fig2.suptitle(r"Screen Distance of Minimum Band Width $\tilde{f}(y_0, \delta y)$") 
           
    # ax2.legend( fontsize="9", loc="upper right")
    # fig2.savefig(f"minWidthDists_1rb.png", dpi=300)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
    # ax2.cla()
    # fig2.clf()
    # plt.close(fig2)
    
#     ax3.errorbar(y0_135rb, xmin_data_135rb, yerr=1, color = 'g', fmt = '-o', ms = 2.5, alpha = 0.75, label = r"Data")
#     ax3.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\tilde{f}$ (mm)')
#     ax3.set_xlim(0.64, 0.9)
#     ax3.set_ylim(-0.2, 10)
#     ax3_x2 = ax3.twiny()
#     # set limits for shared axis
#     ax3_x2.set_xlim(0.64/0.65,0.9/0.65)
#     ax3_x2.set_xlabel(r"$y_0/r_b$")
#    # ax3_x2.xaxis.set_label_coords(0.5, 1.05)
#     fig3.suptitle(r"Screen Distance of Minimum Band Width $\tilde{f}(y_0, \delta y)$") 
          
#     ax3.legend(fontsize="10", loc="upper left")
#     fig3.savefig(f"minWidthDists_overRb.png", dpi=300)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
#     ax3.cla()
#     fig3.clf()
#     plt.close(fig3)
    
#     ax5.errorbar(y0, xmin_data, yerr=1, color = 'g', fmt = '-o', ms = 2.5, alpha = 0.75, label = r"Data")
#     ax5.plot(y0_09rb, xmin_prediction2_5, '--k', linewidth=1, label = 'Eq. [5]') 
#     ax5.plot(y0_09rb, xmin_prediction2_5_2, '--b', linewidth=1, label = 'Eq. [6]') 
#     ax5.plot(y0_09rb, xmin_prediction3_5_3, '--r', linewidth=1, label = 'Eq. [7]') 
#     ax5.axvline(0.36, 0, 1, color = 'magenta', ls = '--', label = r'$y_0 = 0.36$ $c/\omega_p$') 
#     ax5.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\tilde{f}$ (mm)')
#     ax5.set_xlim(-0.02, 0.9)
#     ax5.set_ylim(-10, 150)
#     ax5_x2 = ax5.twiny()
#     # set limits for shared axis
#     ax5_x2.set_xlim(-0.02/0.65,0.9/0.65)
#     ax5_x2.set_xlabel(r"$y_0/r_b$")
#    # ax3_x2.xaxis.set_label_coords(0.5, 1.05)
#     fig5.suptitle(r"Screen Distance of Minimum Band Width $\tilde{f}(y_0, \delta y)$") 
          
#     ax5.legend( fontsize="9", loc="upper right")
#     fig5.savefig(f"minWidthDists_allRb.png", dpi=300)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
#     ax5.cla()
#     fig5.clf()
#     plt.close(fig5)
    
#     ax8.errorbar(y0, xmin_data, yerr=1, color = 'g', fmt = '-o', ms = 2.5, alpha = 0.75, label = r"Data")
#     ax8.plot(y0_09rb, xmin_pred_exp1, '--k', linewidth=1, label = 'Eq. [5]') 
#     ax8.plot(y0_09rb, xmin_pred_exp2, '--b', linewidth=1, label = 'Eq. [6]') 
#     ax8.plot(y0_09rb, xmin_pred_exp3, '--r', linewidth=1, label = 'Eq. [7]')  
#     ax8.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\tilde{f}$ (mm)')
#     ax8.set_xlim(-0.02, 0.9)
#     ax8.set_ylim(-10, 150)
#     ax8_x2 = ax8.twiny()
#     # set limits for shared axis
#     ax8_x2.set_xlim(-0.02/0.65,0.9/0.65)
#     ax8_x2.set_xlabel(r"$y_0/r_b$")
#    # ax3_x2.xaxis.set_label_coords(0.5, 1.05)
#     fig8.suptitle(r"Screen Distance of Minimum Band Width $\tilde{f}(y_0, \delta y_{exp})$") 
    
#     ax8.legend( fontsize="10", loc="upper right")
#     fig8.savefig(f"minWidthDists_allRb_using_experimental-dy.png", dpi=300)#(f"minWidthDists-sampling{mEdge[fileIndex]}Points.png", dpi=300)
#     ax8.cla()
#     fig8.clf()
#     plt.close(fig8)
    
    # ax4.errorbar(y0, minWidths, fmt = '-og', ms = 1.8, label = "Data")
    # #ax4.errorbar(y0, minWidths, fmt = '-ob', ms = 1.8, label = "Data")
    # #ax4.plot(y0, minWidth_prediction, '--k', linewidth=1.2, label = 'Theoretical')
    # ax4.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\sigma_y(\tilde{f}, y_0, \delta y)$ ($c/\omega_p$)')
    # #ax4.set_xlim(0.0, 0.35)
    # ax4.set_ylim(-0.002, 0.1)
    # ax4_x2 = ax4.twiny()
    # # set limits for shared axis
    # #ax4_x2.set_xlim(0.0/0.65,0.9/0.65)
    # ax4_x2.set_xlabel(r"$y_0/r_b$")
    # fig4.suptitle(r"Minimum Band Widths $\sigma_y(\tilde{f}, y_0, \delta y)$")
    # ax4.legend()
    # fig4.savefig(f"minWidths.png", dpi=300)
    # ax4.cla()
    # fig4.clf()
    # plt.close(fig4)
    
    #xmin_avg = np.average(xMinTot, axis = 0)
    #xmin_err = np.sqrt(len(dataF))/len(dataF)
    
    # ax6.errorbar(y0, initWidths, yerr=0.004, fmt = '-og', ms = 0.8, label = "Measured")
    # ax6.axhline(0.048, 0, 1, color = 'k', ls = '--', linewidth=1.2, label = 'Theoretical')
    # ax6.set(xlabel = r'$y_0$ ($c/\omega_p$)', ylabel = r'$\sigma_y(0, y_0, \delta y)$ ($c/\omega_p$)')
    # ax6.set_xlim(-0.02, 0.65)
    # ax6.set_ylim(-0.002, 0.1)
    # fig6.suptitle(r"Initial Band Widths $\sigma_y(\tilde{f}, y_0, \delta y)$")
    # ax6.legend()
    # fig6.savefig(f"initialBandWidths.png", dpi=300)
    # ax6.cla()
    # fig6.clf()
    # plt.close(fig6)


    if (plotFocals): 
        #print(fFile)
        focalData = pd.read_csv(f'{fFile}', sep=',', header = None, index_col=None, engine = "python")
        #print("Data file", minWidthData)
        y0_ = np.array(focalData[0])
        topFocals = np.array(focalData[1])
        bottomFocals = np.array(focalData[2])
        yt_e = np.array(focalData[3])
        yb_e = np.array(focalData[4])
        
        focTheory_top = np.array(focalLength(yt_e))
        focTheory_bottom = np.array(focalLength(yb_e))
        
        ax9.errorbar(y0_, topFocals, yerr=1, fmt = 'ob', ms = 1.8, label = r"Measured $f(y_0+\delta y)$")
        ax9.errorbar(y0_, bottomFocals, yerr=1, fmt = 'sr', ms = 1.8, label = r"Measured $f(y_0)$")
        ax9.plot(y0_, focTheory_top, color = 'k', ls = '--', label = r'Analytic $f(y_0+\delta y)$')
        ax9.plot(y0_, focTheory_bottom, color = 'crimson', ls = '--', label = r'Analytic $f(y_0)$')
        #ax9.axhline(0.05, 0, 1, color = 'k', ls = '--', linewidth=1.2, label = 'Theoretical')
        ax9.set(xlabel = r'$y_0$ [$c/\omega_p$]', ylabel = r'$x$ [mm]')
        ax9.set_xlim(-0.04, 0.65)
        ax9.set_ylim(0, 110)
        fig9.suptitle(r"Experimentally Measured Focal Lengths")
        ax9.legend( fontsize="9")
        fig9.savefig(f"measuredFocalLengths.png", dpi=300)


def singlePointAnalysis(singleFocusingData): 
    
    fig1, ax1 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    fig2, ax2 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    fig3, ax3 = plt.subplots(1,figsize=(6, 4), dpi=300) 
    
    def trueFocal(y0, px, py): 
        
        conversion = 1/6
        f = -np.multiply(y0, np.divide(px,py))*conversion
        
        return f
    
    def geometricFocal(y0, px, py, pz): 
        
        p2 = np.square(px) +  np.square(py) +  np.square(pz)
        gamma = np.sqrt(p2)
        k = 0.475
        rb = 0.65
        conversion = 1/6 
        f0 = np.divide(p2, 2*gamma*k*rb)*conversion
        f = f0*np.divide(1, np.sqrt(1-np.square(y0/rb)))
        
        return f
    
    singleFocusData = pd.read_csv(f'{singleFocusingData}', sep=',', header = None, index_col=None, engine = "python")
    y0_sim = np.array(singleFocusData[0])
    ytop = np.array(singleFocusData[1])
    focus_top = np.array(singleFocusData[2])
    px_top, py_top, pz_top = np.array(singleFocusData[3]), np.array(singleFocusData[4]), np.array(singleFocusData[5])
    ybot = np.array(singleFocusData[6])
    focus_bottom = np.array(singleFocusData[7])
    px_bot, py_bot, pz_bot = np.array(singleFocusData[8]), np.array(singleFocusData[9]), np.array(singleFocusData[10])
    
    true_focus_top = np.array(trueFocal(ytop, px_top, py_top))
    true_focus_bot = np.array(trueFocal(ybot, px_bot, py_bot))
    approx_focus_top = np.array(geometricFocal(ytop, px_top, py_top, pz_top))
    approx_focus_bot = np.array(geometricFocal(ybot, px_bot, py_bot, pz_bot))
    
    # ax1.errorbar(y0_sim, focus_top, yerr=1, fmt = 'ob', ms = 2.5, label = r"Measured $f(y_0+\delta y)$")
    # ax1.plot(y0_sim, true_focus_top, color = 'crimson', ls = '--',label = r"Eq.[2] ($y_0+\delta y$)")
    # ax1.plot(y0_sim, approx_focus_top, color = 'k', ls = '--',label = r"Eq.[1] ($y_0+\delta y$)")
    # ax1.set(xlabel = r'$y_0$ [$c/\omega_p$]', ylabel = r'$x$ [mm]')
    # ax1.set_xlim(-0.04, 0.65)
    # ax1.set_ylim(0, 110)
    # fig1.suptitle(r"Focal Lengths at the Top of Band")
    # ax1.legend(fontsize="9")
    # fig1.savefig(f"measuredFocalLenghts_top.png", dpi=300)
    
    # ax2.errorbar(y0_sim, focus_bottom, yerr=1, fmt = 'or', ms = 2.5, label = r"Measured $f(y_0)$")
    # ax2.plot(y0_sim, true_focus_bot,color = 'indigo', ls = '--',label = r"Eq.[2] ($y_0$)")
    # ax2.plot(y0_sim, approx_focus_bot,color = 'darkgreen', ls = '--',label = r"Eq.[1] ($y_0$)")
    # ax2.set(xlabel = r'$y_0$ [$c/\omega_p$]', ylabel = r'$x$ [mm]')
    # ax2.set_xlim(-0.04, 0.65)
    # ax2.set_ylim(0, 110)
    # fig2.suptitle(r"Focal Lengths at the Bottom of Band")
    # ax2.legend(fontsize="9")
    # fig2.savefig(f"measuredFocalLenghts_bottom.png", dpi=300)
    
    ax3.errorbar(y0_sim, focus_top, yerr=1, fmt = 'ob', ms = 1.8, label = r"Measured $f(y_0+\delta y)$")
    ax3.errorbar(y0_sim, focus_bottom, yerr=1, fmt = 'or', ms = 2.5, label = r"Measured $f(y_0)$")
    ax3.plot(y0_sim, approx_focus_top, color = 'k', ls = '--', label = r"Analytic $f(y_0+\delta y)$")
    ax3.plot(y0_sim, approx_focus_bot, color = 'darkgreen', ls = '--', label = r"Analytic $f(y_0)$")
    #ax9.axhline(0.05, 0, 1, color = 'k', ls = '--', linewidth=1.2, label = 'Theoretical')
    ax3.set(xlabel = r'$y_0$ [$c/\omega_p$]', ylabel = r'$x$ [mm]')
    ax3.set_xlim(-0.04, 0.65)
    ax3.set_ylim(0, 110)
    fig3.suptitle(r"Experimental Focal Lengths")
    ax3.legend(fontsize="9")
    fig3.savefig(f"measuredFocalLenghts_together.png", dpi=300)
    