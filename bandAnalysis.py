from logging import raiseExceptions
import sys
import os
import time
import importlib
import numpy as np
import tqdm
import pickle
from DebugObjectModule import DebugObject
from random import randint
import multiprocessing as mp

import include.makeFullAnimation_dynMasks as makeFullAni
import include.writeFullEvolData as writeHist
import include.weighting_masks_function_rprism_dynamic as weightmaskFunc
import include.plotWeights_dynMasks as plotWeights
import include.saveBandPoints as saveBP

# Be sure to change .npz file name location from main.py output!
# Put .npz file in /data directory

# Weighting Options (Only applicable for showFullEvolution and makeFullAnimation plot):
useWeights_x = False                 # NOT CURRENTLY IN USE - LEAVE FALSE - Use weights in x-direction
useWeights_y = True#False                  # Use gaussian weights in y-direction
useWeights_xi = True#False                 # Use gaussian weights in xi-direction

skipWeightingCalc = False            # Skip weighting calculation and use imported pre-calculated weights
saveWeights = False                 # Save weights to .npz file (Remember to move to ./data directory!)

# Masking Options:
useMasks_xi = False                 # Use masks in xi-direction (Vertical; done during weighting)
useMasks_y = True#False                  # Use masks in y-direction (Horizontal; done during weighting)
useMasks_x = False                  # NOT CURRENTLY IN USE - LEAVE FALSE - Use masks in x-direction (transverse; done during weighting)

useCurtainMask_y = False                   #Create a mask where only one side moves continuously
useMovingMask_y = False                    #Create a mask that continuously moves
useMovingBand_y = True#False                    #Create a thin band of electrons that moves continuously (inverse of moving mask)

useCurtainMask_z = False                   #Create a mask where only one side moves continuously
useMovingMask_z = False                    #Create a mask that continuously moves
useMovingBand_z = False

# Plotting Scripts
# View full evolution of probe at hardcoded locations in colored histograms # Use for high density probes

saveSinglePoints = False#True
plotXYtraj = False#True
calcBandWidths = False#True

makeFullAnimation = False#True
plotTruncatedBandData = False
separateMaxVals = False#True

singlePointAnalysis = False#True
calcDerivs = True
plotFocals = False#True

plotBandWidthEvo = False#True
plotMinBandWidths = False#True

# Gaussian Weighting Testing
plotWeightsx = False                  # Plot w vs xi (ONLY for single line of particles in x-dir)
plotWeightsy = False                  # Plot w vs y (ONLY for single line of particles in y-dir)
plotWeightsxi = False                  # Plot w vs y (ONLY for single line of particles in xi-dir)
plotWeightsxiy = False                 # Plots initial particle density map
 
nMasks = 1#29#45 
y0_init = 0.0
mask_step = 0.02                             #initial height of lower edge of mask
dy_mask = 0.0+y0_init                                 
dz_mask = 0.0 
mask_bools = np.array([useCurtainMask_y,useMovingMask_y,useMovingBand_y,useCurtainMask_z,useMovingMask_z,useMovingBand_z])
band_heights = np.linspace(y0_init, nMasks*mask_step, nMasks+1)

if __name__ == '__main__':

    numberOfCores = 1#8 #mp.cpu_count() 
    print(f"Number of cores used for multiprocessing: {numberOfCores}")
    pool = mp.get_context('spawn').Pool(numberOfCores)

    start_time = time.time()
    t = time.localtime()
    curr_time = time.strftime("%H:%M:%S", t)
    print("index.py - START TIME: ", curr_time)
    
    if (len(sys.argv) >= 2):

        # Get inital conditions of probe again
        input_fname_1 = str(sys.argv[1])
        print("Using initial conditions from ", input_fname_1)
        init = importlib.import_module(input_fname_1)
        sim_name = init.simulation_name
        shape_name = init.shape
        xden = init.xdensity
        yden = init.ydensity
        xiden = init.xidensity
        res = init.resolution
        iter = init.iterations
        mode = init.mode
        fname = init.fname
        debugmode = init.debugmode
        x_c = init.x_c
        y_c = init.y_c
        xi_c = init.xi_c
        px_0 = init.px_0
        py_0 = init.py_0
        pz_0 = init.pz_0
        x_s = init.x_s
        s1 = init.s1
        s2 = init.s2

        if len(sys.argv) == 3:
            # Get initial conditions of beam
            input_fname_2 = str(sys.argv[2])
            print("Using beam conditions from ", input_fname_2)
            beaminit = importlib.import_module(input_fname_2)
            beamx_c = beaminit.beamx_c
            beamy_c = beaminit.beamy_c
            beamxi_c = beaminit.beamxi_c
            sigma_x=beaminit.sigma_x
            sigma_y=beaminit.sigma_y
            sigma_xi=beaminit.sigma_xi
        else:
            print("WARNING: No gaussian weights inputted. Make sure not using weighting!")

        # Load data from npz file export from main.py
        data = np.load('./data/' + fname) # Change this line as needed
        x_0 = data['x_init']
        y_0 = data['y_init']
        xi_0 = data['xi_init']
        z_0 = data['z_init']
        x_f = data['x_dat']
        y_f = data['y_dat']
        xi_f = data['xi_dat']
        z_f = data['z_dat']
        px_f = data['px_dat']
        py_f = data['py_dat']
        pz_f = data['pz_dat']
        t0 = data['t_dat']
        
        print("Data loaded!")

        noObj = len(x_0) # Number of particles in the simulation (2D Projection)

        for N in range(0, nMasks):
            
            y0_mask = band_heights[N] 
        
            # WEIGHTING IMPORTS/SAVING
            rand = "{:02d}".format(randint(0,99))
            weights_fname = fname[:-4] + "-weights-" + rand
            if (skipWeightingCalc):
                print("Not using weighting!")
                #Do not use these when simulating probe with no weights - i.e. for plot2Dtracks
                data = np.load('./data/' + weights_fname + '.npz') # Change this line as needed
                w = data['w']
                print(f"\nUsing weights from {'./data/' + weights_fname + '.npz'}...\n")
            else:
                # Create weighting array with appropriate masks
                w = []
                w = [1 for k in range(0,noObj)] #Creates default array of weights with length noObj, each with value 1
                        
                start_time_w = time.time()
                t_w = time.localtime()
                curr_time_w = time.strftime("%H:%M:%S", t_w)
                print("\nWeighting calculations - START TIME: ", curr_time_w)

                # Call weighting function getWeights 
                # Note: w_virt, xv, yv, xiv, only used for debugging purposes
                
                w, w_export1, w_y, w_xi, topM, bottomM, leftM, rightM = weightmaskFunc.getWeights(beamx_c,beamy_c,beamxi_c,x_c,y_c,xi_c,s1,s2,xden,yden,xiden,res,sigma_x,sigma_y,sigma_xi,noObj,t0,useWeights_x,useWeights_y,useWeights_xi,useMasks_x,useMasks_xi,useMasks_y, mask_bools, dy_mask, dz_mask)    
                
                t_w_end = time.localtime()
                curr_time_w_end = time.strftime("%H:%M:%S", t_w_end)
                print("Weighting calculations - END TIME: ", curr_time_w_end)
                print("Weighting calculations - DURATION: ", (time.time() - start_time_w)/60, " min\n")
                
                if (saveWeights):
                    np.savez(weights_fname, w=w)
                    print(f"\nWeights saved to {weights_fname + '.npz'}\n") #Saves weights for reuse
                    
                print(f"Top and bottom of band located at: y={topM}, y={bottomM}")

            if useMasks_y != True: 
                assert useMasks_y == True, "Need to make an electron band to use program!"

            if useMovingBand_y != True: 
                assert useMovingBand_y == True, "Need to have a dynamic electron band to use program!"
            
            if (plotWeightsxiy):
                
                plotWeights.plotweightsxiy(y_0,xi_0, w, rand, mask_bools, topM, bottomM, leftM, rightM, N)
                
                print(f"Initial probe weighting plotted for y0 = {y0_mask}!")
            
            # # Normal Plotting of Band Progression
            if (makeFullAnimation):
                
                plasma_bnds, slices, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, cmap, cmin, vmin_, vmax_, zmin, zmax, ymin, ymax, fps, new_path, screen_dists = makeFullAni.prepare(sim_name, shape_name, noObj, rand, N)
                # Multiprocessing: propagate to each screen and create frame
                start_time_pfc = time.time()
                t_pfc = time.localtime()
                curr_time_pfc = time.strftime("%H:%M:%S", t_pfc)
                print("Multiprocessing propagation and frame creation - START TIME: ", curr_time_pfc)

                pool.starmap(makeFullAni.plotmp,[(i,xi_f,x_f,y_f,z_f,px_f,py_f,pz_f, w, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, cmap, cmin, vmin_, vmax_, zmin, zmax, ymin, ymax, new_path, screen_dists, mask_bools, topM, bottomM, leftM, rightM, N) for i in range(0,slices)])

                #pool.close()
                
                #pool.join()

                t_pfc_end = time.localtime()
                curr_time_pfc_end = time.strftime("%H:%M:%S", t_pfc_end)
                print("MP PFC - END TIME: ", curr_time_pfc_end)
                print("MP PFC - DURATION: ", (time.time() - start_time_pfc)/60, " min\n")
                
                
            print("Truncating dataset to perform band-width evolution calculations . . .")
            
            plasma_bnds, slices, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists = saveBP.prepare(sim_name, shape_name, noObj, rand, y0_mask)

            bandX, bandY, bandZ, bandW, bandPx, bandPy, bandPz = saveBP.getBandData(0,x_f,y_f,z_f,px_f,py_f,pz_f, w, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM)
            
            if (saveSinglePoints): 
                
                singleData = 'initialPointsData.txt' #file to save momentum data
                
                #singleData = str(singleData)
                #print(f"Saving width data to {pFileName}")
                if N == 0:
                    if os.path.isfile(singleData): 
                        os.system(f'rm {singleData}') 
                
                file = open(singleData, 'a')
                
                ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, ybot_0, zbot_0, pxbot_0, pybot_0, pzbot_0  = saveBP.individualParticleExtremumData(0,bandX,bandY,bandZ,bandPx,bandPy,bandPz, y0_mask, new_path, singleData, screen_dists, topM, bottomM)
            
                file.close()
                
                print('Single point for top and bottom of band saved for analysis.')
                
            
            #insert band width centering code here
            if (separateMaxVals): 
                
                print("Number of screens = ", slices)
                
                print("Locating calibration points for finding max and min band heights . . .")
                
                pool.starmap(saveBP.getCalibrationPoints,[(i, bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM)for i in range(0,slices)])
            
            if (plotTruncatedBandData):
                
                start_time_bandPlot = time.time()

                print("Number of screens = ", slices)

                pool.starmap(saveBP.plotBands,[(i, bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, xden, plasma_bnds, xs_norm, yslice, zslice, bin_edges_z, bin_edges_y, zmin, zmax, ymin, ymax, new_path, screen_dists, topM, bottomM, ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, ybot_0, zbot_0, pxbot_0, pybot_0, pzbot_0) for i in range(0,slices)])

                #pool.close()

                #pool.join()

                print("Band Plotting Duration: ", (time.time() - start_time_bandPlot)/60, " min\n")
            
            #insert width verification plotting here
                
            if (calcBandWidths):

                include_plots = True
                
                dataFileName = f'bandletWidthData_y0-{y0_mask:.02f}'.replace(".","")+'.txt'
                
                dataFileName = str(os.path.join(new_path,dataFileName))
                print(f"Saving width data to {dataFileName}")
                
                file = open(dataFileName, 'w')
                
                pFileName = f'bandletMomentumData_y0-{y0_mask:.02f}'.replace(".","")+'.txt' #file to save momentum data
                
                pFileName = str(os.path.join(new_path,pFileName))
                #print(f"Saving width data to {pFileName}")
                
                file = open(pFileName, 'w')

                print("Number of screens = ", slices)
                
                pool.starmap(saveBP.propagateBandlet,[(i, bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, plasma_bnds, xs_norm, yslice, zslice, screen_dists, topM, bottomM, dataFileName, pFileName,new_path, include_plots) for i in range(0,slices)])

                file.close()

                #pool.close()

                #pool.join()

            #now do focusing analysis
            if (saveSinglePoints):
                
                singleData = 'singlePointData.txt' 
                
                singleData  = str(os.path.join(new_path,singleData))
                #print(f"Saving width data to {pFileName}")
                
                file = open(singleData, 'a')
                
                pool.starmap(saveBP.saveSinglePointTrajectory,[(i, bandX, plasma_bnds, xs_norm, new_path, screen_dists, y0_mask, ytop_0, ztop_0, pxtop_0, pytop_0, pztop_0, ybot_0, zbot_0, pxbot_0, pybot_0, pzbot_0, singleData) for i in range(0,slices)])

                file.close()
                
            if (plotXYtraj):
                print('Plotting X-Y trajectory.')
                
                singleData = 'singlePointData.txt' 
                singleData  = str(os.path.join(new_path,singleData))
                
                singleData2 = 'singlePointFocalLengths.txt'
                if N == 0:
                    if os.path.isfile(singleData2): 
                        os.system(f'rm {singleData2}') 
                
                file = open(singleData, 'r')        
                file2 = open(singleData2, 'a')
                
                saveBP.saveSingleFocusing(new_path,y0_mask,pxtop_0, pytop_0, pztop_0, pxbot_0, pybot_0, pzbot_0, singleData, singleData2)
                
                file.close()
                file2.close()
                
            if (calcDerivs): 
                
                print("Calculating f'(y0) and f''(y0) . . .")
                
                saveBP.calculateDerivatives(bandX,bandY,bandZ,bandPx,bandPy,bandPz, bandW, plasma_bnds, xs_norm, yslice, zslice, screen_dists, topM, bottomM, new_path)


            if (plotBandWidthEvo):
                # # Plot Band Widths according to formula below

                txtpath = new_path #f'{foldName}/bandWidthDataFiles/'
                txtpath = str(txtpath)
                dataFiles = os.listdir(txtpath)
                widthEvoFiles = []
                for iFile, files in enumerate(dataFiles):
                    # check the files which have a specific name
                    if "Width" in files:
                        # print path name of selected files
                        widthEvoFiles.append(files)
                        
                txtpath2 = str(new_path) #f'{foldName}/bandWidthDataFiles/'
                dataFiles2 = os.listdir(txtpath2)
                momFiles = []
                for iFile, files in enumerate(dataFiles2):
                    # check the files which have a specific name
                    if "Momentum" in files:
                        # print path name of selected files
                        momFiles.append(files)
                
                #widthEvoFiles = np.sort(widthEvoFiles)
                focalLengthsFile = f'focal_data.txt'
                xminFile = f'xmin_dy-0p048.txt'
                
                xFile =  open(xminFile, 'a')
                fFile = open(focalLengthsFile, 'a')

                saveBP.plotWidthEvo(new_path, xminFile, widthEvoFiles[0], momFiles[0], y0_mask, plotFocals, focalLengthsFile)
                
                xFile.close()
                fFile.close()
                      
            #pool.close()

            print(f'Iteration number {N+1} complete!')
            dy_mask += mask_step
            y0_mask = dy_mask
            print(f"Moving top edge of mask to y = {dy_mask}")

    
        if (plotMinBandWidths):
            
            print("Plotting xmin data . . .")
            xminFile = f'xmin_dy-0p048.txt'
            xFile =  open(xminFile, 'r')
            
            if (plotFocals): 
                print("Plotting focal lengths . . .")
                
            focalLengthsFile = f'focal_data.txt'
            fFile = open(focalLengthsFile, 'r')
            #print(fFile)

            saveBP.plotMinBandWidths(xminFile, focalLengthsFile, plotFocals)  
            
            xFile.close()
            fFile.close()
            
            #os.system('mv minWidthDists*.png xminData_dy010/')
            
            print("Minimum width positions plotted!")  
        
        if (singlePointAnalysis): 
            
            print('Performing analysis from saving one set of points . . .')
            
            singleFocusingData = 'singlePointFocalLengths.txt'
            
            file = open(singleFocusingData, 'r')
            
            saveBP.singlePointAnalysis(singleFocusingData)
            
            file.close()
            
            
    pool.close()

    pool.join()

