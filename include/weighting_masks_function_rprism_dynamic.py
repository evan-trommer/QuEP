# For use with shape files of 'rprism' only

import sys
import math
import numpy as np
import pdb
import progressbar
import time

# Creates weights based on distribution and inputted masks (below)

def getWeights(beamx_c,beamy_c,beamxi_c,x_c,y_c,xi_c,s1,s2,xdensity,ydensity,xidensity,resolution,sigma_x,sigma_y,sigma_xi,noObj,t0,useWeights_x,useWeights_y,useWeights_xi,useMasks_x,useMasks_xi,useMasks_y, mask_bools, dy_mask, dz_mask):
    
# Recompute necessary parameters (as done in shape file)
    xistep = 2*s2/xidensity # Can also use resolution here
    ystep = 2*s1/ydensity
    xstep = xistep          # Purposefully setting xstep as equal to xistep for projection of x onto xi

# Calculate s3 
    s3 = xstep*(xdensity-1)/2.0
    
# Define corners (xfront is first to enter field)
    ytop = y_c + s1
    ybot = y_c - s1
    xileft = xi_c - s2
    xiright = xi_c + s2
    xfront = x_c + s3  
    xback = x_c - s3
    
# Start in front top right
    yn = ytop
    xin = xiright
    xn = xfront
    zn = xiright + t0
    
#specialty masking
    useCurtainMask_y = mask_bools[0]
    useMovingMask_y = mask_bools[1]   
    useMovingBand_y = mask_bools[2]
    useCurtainMask_z = mask_bools[3]
    useMovingMask_z = mask_bools[4]
    useMovingBand_z = mask_bools[5]
    
    if (useMasks_y or useMasks_xi):
        print(f'Curtain masking in y = {useCurtainMask_y}; Moving masking in y = {useMovingMask_y}; Moving y-band = {useMovingBand_y}; .')
        print(f'Curtain masking in z = {useCurtainMask_z}; Moving masking in z = {useMovingMask_z}; Moving z-band = {useMovingBand_z}; .')
    
    print("Creating weighting arrays...")
# Create individual coordinate arrays    
    x_0 = np.linspace(xfront,xback,xdensity)      # List of all possible inital x  positions of all particles going through simulation
    y_0 = np.linspace(ytop,ybot,ydensity)         # List of all possible inital y  positions of all particles going through simulation
    xi_0 = np.linspace(xileft,xiright,xidensity)  # List of all possible inital xi positions of all particles going through simulation
    
# Masking and weighting for y and xi --------------------------------------------------------------------
    
    # Default w_y and w_xi weights
    w_y = np.full(y_0.shape, 1.0)
    w_xi = np.full(xi_0.shape, 1.0)

    topM, bottomM, leftM, rightM = 0.0, 0.0, 0.0, 0.0
    
    if (useWeights_y): # If using gaussian weighting in y, apply to w_y
        w_y = np.exp((-1.*(y_0-beamy_c)**2)/(2*sigma_y**2)) # Calculate weights for each y slice
    
    if (useMasks_y): # If using Masks for y, apply them to y weighting array
        w_y, topM, bottomM = yMasks(y_0,w_y,useCurtainMask_y, useMovingMask_y, useMovingBand_y, dy_mask)
        
    if (useWeights_xi): # If using gaussian weighting in xi, apply to w_xi
        w_xi = np.exp((-1.*(xi_0-beamxi_c)**2)/(2*sigma_xi**2)) # Calculate weights for each xi slice

    if (useMasks_xi): # If using Masks for xi, apply them to xi weighting array
        w_xi, leftM, rightM = xiMasks(xi_0,w_xi, useCurtainMask_z, useMovingMask_z, useMovingBand_z, dz_mask)

    w_export1 = []
    w_export2 = w_y
    w_export3 = w_xi
    
    # Create empty weighting list
    w = []
    w = [0 for k in range(0,noObj)] # Creates weighting array of length noObj, with default value 0
    
# Loop through x layers to calculate weights with masks and add to 2D Y-Xi projection
    for i in progressbar.progressbar(range(0,len(x_0)), redirect_stout=False):
        start_time_weightcalc = time.time()

        if (useWeights_x):
            # Find weight value for this x-slice
            w_x = np.exp((-1.*(x_0[i]-beamx_c)**2)/(2*sigma_x**2)) # Gives a float

            # Check for x mask
            w_x = xMasks(useMasks_x,x_0[i],w_x)

            w_export1.append(w_x)

            w_y = w_y * w_x

        # Create final weighting list w to return
        # Maps 3d virtual particles in x-layer onto 2d Y-Xi projection appropriate location           
        for k in range(0, len(w_xi)):
            # Multiply by xi weighting if in use
            w_virt = w_y * w_xi[k]
            
            # Loop through y layers and apply weighting in appropriate location
            for j in range(0,len(y_0)):
                w[(xidensity * j + k) + ydensity*xidensity*i] += w_virt[j]

        # Delete/Deallocate arrays for memory conservation
        w_x = None
        w_virt = None
        w_xy = None
    
    w_export1 = np.array(w_export1)
    w_export4 = w[int(xidensity*(ydensity/2)):int(xidensity*(ydensity/2)+xidensity)]
    #np.savez("weight-exports.npz", w_exportx=w_export1, w_exporty=w_export2, w_exportxi=w_export3, w_exportvirt=w_export4)
    return w, None, None, None, topM, bottomM, leftM, rightM

def xiMasks(xi_0, w_xi, useCurtainMask_z, useMovingMask_z, useMovingBand_z, dz_mask):
    
    if (useCurtainMask_z):
        rightZ = -4.5
        leftZ = rightZ - dz_mask
        # Define masks in xi direction. Change if different mask is desired
        left_of_masks = [leftZ]  # left most limit of each mask in order, on inital xi position
        right_of_masks = [rightZ]  # right most limit of each mask in order, on initial xi position
        leftM = left_of_masks[0]
        rightM = right_of_masks[0]
        # Apply masks to w_xi
        for g in range(0,len(left_of_masks)):
            w_xi = np.where(np.logical_and(xi_0 > left_of_masks[g], xi_0 < right_of_masks[g]), 0, w_xi)
    
    elif (useMovingMask_z):
        rightZ = -4.5 - dz_mask
        leftZ = rightZ - 0.5 #mask of width 0.5
        # Define masks in xi direction. Change if different mask is desired
        left_of_masks = [leftZ]  # left most limit of each mask in order, on inital xi position
        right_of_masks = [rightZ]  # right most limit of each mask in order, on initial xi position
        leftM = left_of_masks[0]
        rightM = right_of_masks[0]
        # Apply masks to w_xi
        for g in range(0,len(left_of_masks)):
            w_xi = np.where(np.logical_and(xi_0 > left_of_masks[g], xi_0 < right_of_masks[g]), 0, w_xi)
    
    elif (useMovingBand_z):
        rightZ = -4.5
        leftZ = rightZ- dz_mask
        rightZ_2 = rightZ-0.5-dz_mask #band of width 0.5c/wp
        # Define masks in xi direction. Change if different mask is desired
        left_of_masks = [leftZ, -14.5]  # left most limit of each mask in order, on inital xi position
        right_of_masks = [rightZ, rightZ_2]  # right most limit of each mask in order, on initial xi position
        leftM = left_of_masks[0]
        rightM = right_of_masks[1]
        # Apply masks to w_xi
        for g in range(0,len(left_of_masks)):
            w_xi = np.where(np.logical_and(xi_0 > left_of_masks[g], xi_0 < right_of_masks[g]), 0, w_xi)
    
    else:
        # Define masks in xi direction. Change if different mask is desired
        left_of_masks = []  # left most limit of each mask in order, on inital xi position
        right_of_masks = []  # right most limit of each mask in order, on initial xi position
        leftM = 0
        rightM = 0
        # Apply masks to w_xi
        for g in range(0,len(left_of_masks)):
            w_xi = np.where(np.logical_and(xi_0 > left_of_masks[g], xi_0 < right_of_masks[g]), 0, w_xi)

    return w_xi, leftM, rightM

def yMasks(y_0, w_y, useCurtainMask_y, useMovingMask_y, useMovingBand_y, dy_mask):    
    
    if (useCurtainMask_y):
        
        top = 1.0
        bottom = 1.0 - dy_mask
        # Define masks in y direction, 0 is 0 on the y-axis. Change if different mask is desired
        top_of_masks = [top]  #upper limit of each mask in order, on inital y position
        bot_of_masks = [bottom] #lower limit of each mask in order, on inital y position
        topM = top_of_masks[0]
        bottomM = bot_of_masks[0]
        
        #Apply masks to w_y
        for h in range(0,len(top_of_masks)):            
            w_y = np.where(np.logical_and(y_0 > bot_of_masks[h], y_0 < top_of_masks[h]), 0, w_y)             
            
    elif (useMovingMask_y):
        
        top = 1.0 - dy_mask
        bottom = top - 0.1    #create mask of thickness 0.1 c/w_p
        top_of_masks = [top]  #upper limit of each mask in order, on inital y position
        bot_of_masks = [bottom] #lower limit of each mask in order, on inital y position
        #print(f"Top of mask is {top_of_masks[0]}, bottom of mask is {bot_of_masks[0]}")
        topM = top_of_masks[0]
        bottomM = bot_of_masks[0]
        
        #Apply masks to w_y
        for h in range(0,len(top_of_masks)):            
            w_y = np.where(np.logical_and(y_0 > bot_of_masks[h], y_0 < top_of_masks[h]), 0, w_y)
            
        
    elif (useMovingBand_y):
        
        top = 1.1
        bottom = 0.048+dy_mask#0.05+dy_mask #   
        top2 = bottom-0.048#bottom-0.05 #             #create band of thickness 0.1 c/w_p
        top_of_masks = [top, top2]     #upper limit of each mask in order, on inital y position
        bot_of_masks = [bottom, -1.0] #lower limit of each mask in order, on inital y     
        topM = bot_of_masks[0]        #band edges
        bottomM = top_of_masks[1]
        
        #Apply masks to w_y
        for h in range(0,len(top_of_masks)):            
            w_y = np.where(np.logical_and(y_0 > bot_of_masks[h], y_0 < top_of_masks[h]), 0, w_y)
            
            
    else: 
        top_of_masks = [1.0,0.0]  #upper limit of each mask in order, on inital y position
        bot_of_masks = [0.1,-1.0] #lower limit of each mask in order, on inital y position
        
        topM = bot_of_masks[0] 
        bottomM = top_of_masks[-1]
        #Apply masks to w_y
        for h in range(0,len(top_of_masks)):            
            w_y = np.where(np.logical_and(y_0 > bot_of_masks[h], y_0 < top_of_masks[h]), 0, w_y)              
        
    return w_y, topM, bottomM

def xMasks(useMasks_x,x_0_current,w_x):
    if (useMasks_x):
        # Define masks in x direction. Change if different mask is desired
        back_of_masks = []  # back limit of each mask in order, on inital x position
        front_of_masks = []  # right limit of each mask in order, on initial x position
        # Apply masks to w_x
        for m in range(0,len(back_of_masks)):
            if (np.logical_and(x_0_current > back_of_masks[m], x_0_current < front_of_masks[m])): # If in region of mask
                w_x = 0 # Set x weight to zero (0)

    return w_x