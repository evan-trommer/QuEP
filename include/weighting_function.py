import sys
import math
import numpy as np
import pdb

# Creates weights based on distribution

def getWeightsX(beamx_c,beamy_c,beamxi_c,x_c,y_c,xi_c,s1,s2,xdensity,ydensity,xidensity,resolution,sigma_x,sigma_y,noPart):

    # Recompute necessary parameters (as done in shape file)
    xistep = 2*s2/xidensity # Can also use resolution here
    xstep = xistep          # Purposefully setting xstep as equal to xistep for projection of x onto xi
    xidensity_ = xidensity + xdensity - 1  # Allows enough particles in xi direction for x layering

    s3 = xstep*(xdensity-1)/2.0

    # Creating weighting array for layers in x
    wxlayers = []
    wxlayers = [0 for k in range(0,xdensity)]
    for m in range(0,xdensity):
        Deltax = (x_c+s3) - m*xstep - beamx_c                      # Calculate distance from center of distribution
        wxlayers[m] = math.exp((-1.*Deltax**2)/(2*sigma_x**2))           # Calculate weight for each layer m in x-direction in new array wx
    
    # Now we can copy weighting array in x to xi, same for every row in y
    w_x = []
    w_x = [0 for k in range(0,noPart)]
    for i in range(0,xdensity):
        for j in range(0,ydensity):
            for k in range(0,xidensity):
                w_x[xidensity_ * j + k + i] += wxlayers[i]

    return w_x

def getWeightsY(beamx_c,beamy_c,beamxi_c,x_c,y_c,xi_c,s1,s2,xdensity,ydensity,xidensity,resolution,sigma_x,sigma_y,noPart):

    # Recompute necessary parameters (as done in shape file)
    ystep = 2*s1/ydensity
    xidensity_ = xidensity + xdensity - 1  # Allows enough particles in xi direction for x layering

    # Creating weighting array for layers in y
    wylayers = []
    wylayers = [0 for k in range(0,ydensity)]
    for n in range(0,ydensity):
        Deltay = (y_c+s1) - n*ystep - beamy_c                      # Calculate distance from center of distribution
        wylayers[n] = math.exp((-1.*Deltay**2)/(2*sigma_y**2))           # Calculate weight for each layer n in y-direction in new array wy
    
    w_y = []
    w_y = [0 for k in range(0,noPart)]
    # For y
    #for j in range(0,ydensity):
    #    for k in range(0,xidensity_):
    #        w_y[xidensity_ * j + k] += wylayers[j]
    for i in range(0,xdensity):
        for j in range(0,ydensity):
            for k in range(0,xidensity):
                w_y[xidensity_ * j + k + i] += wylayers[j]

    return w_y