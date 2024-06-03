# QuEP

This code was developed to simulate an electron probe's trajectory through a plasma wakefield accelerator. It does so by taking previously simulated fields and propagating an electron through these fields. This allows us to quickly estimate the trajectories of multiple electrons without the repeated use of computationally expensive simulations.

The `master` branch contains the most up-to-date code.

### Running the Simulation
To run the simulation, add an initialization file in the `input` folder, such as the file `example.py`, which would be input to the simulation using
```
python main.py input.example
```

The following parameters must be specified in the initialization file (case insensitive):

* simulation_name - `quasi3D` only (for now!)
* shape - `hline`, `vline`, `single`, `rectangle`, `rprism`, or `rprism_weighted_after`
* iterations - Maximum number of steps before tracking in plasma is stopped
* mode - An extra flag to be used by any simulation. See simulation files under `/include/simulations` for specifications. 
* fname - File name
* x_c, y_c, xi_c - Center coordinates of probe 
* px_0, py_0, pz_0 - Initial momenta of probe
* x_s - Screen distances. Used in certain plotting scripts.
* s1, s2, s3 - Shape parameters. See initialization files under `/include/shape` for specifications. 
* ydensity, xidensity - Number of electrons to be distributed in specified direction. See initialization files under `/include/shape` for specifications.
* resolution - An unused parameter that can refer to the spacing between electrons, if `shape` initialization files are modified.
* debugmode - A boolean parameter that determines whether data of the particle from within the wakefield is collected and exported. Should only be `True` for `single` shape.

Initialization conditions, especially starting coordinates, should be chosen based on the wakefield feature one wants to observe. More conditions may be added by including a new variable in the initialization file, then adding another argument to the `main.py` file (~Line 55), as well as plotting scripts, as deemed necessary. 

A `.npz` file that includes the final x, y, xi, and z positions of each electron, as well as the final x, y, and z momenta, will then be saved.

### Plotting Results

To plot generated data, set the appropriate plotting script macros' booleans to `True` in `index-mp.py` (be sure to check the file name readout location), then run

```
python index-mp.py input.example
```

If you are using weights for the probe, you must create a weighting input file such as `example-weights.py` and input the beam's initial center coordinates, `beamx_c`, `beamy_c`, and `beamxi_c`, as well as its spread parameters, `sigma_x`, and `sigma_y`, `sigma_xi`. Then run, 

```
python index-mp.py input.example input.example-weights
```

If you are plotting using multiple masks (or "dynamic masks") then use the `bandAnalysis.py` file to do post-processing analysis. The appropriate plotting macros are included in this file. This will eventually be merged into a single file. 

The scripts for putting images into an animation are temporarily disabled in `bandAnalysis.py`. For now, it is best to run `movieWriter.py` separately.  

Plotting scripts MUST be edited based on your probe initial conditions, otherwise you might not see anything!

### Masking Procedures 

There are two main masking options: using longitudinal (xi direction) masks, or vertical (y direction) masks. Ability to mask in the transverse (x direction) has not yet been fully implemented, so *always* leave `useMasks_x` as `FALSE` until further notice. 

For both masking directions, there are three types of dynamic masking protocals you can call: curtain masks, moving masks and moving bands. The curtain mask script creates a mask with one edge that continuously moves in a specified direction. The moving mask has both edges of the mask dynamically update as the code runs to create a masking band whose initial position changes with each iteration of the code. The moving band creates two masks, a higher and lower mask, where the higher mask has it's lower edge updated with each iteration, and the lower mask has its upper edge updated. It is important that you only use one of these procedures at a time. Note also that these procedures only work for a **single** mask - if you wish to use multiple masks (e.g. to simulate a grid pattern) you will need to update `weighting_masks_function_rprism_dynamic.py`.

The two numerical variables that are important while doing dynamic masking is `mask_step` and `nMasks`. `mask_step` gives the step size for changing the mask. `nMasks` tells the code how many times to loop the plotting scripts, and should be set appropriately to get the number of different masking configurations. E.g. if you want to create a curtain mask whose edge starts at y=0.9 and you want to bring it down to y=0.0 in steps of 0.1, set `mask_step = 0.1` and `nMasks = 10`.

While using the dynamic masking procedures, the only plotting boolean values that should be set to TRUE in `bandAnalysis.py` are `makeFullAnimation` and the appropriate weight plotting boolean.

### Repository Structure 

The `master` branch contains the most up-to-date code.

The `version-2.0` branch contains the second complete version of the code, as left by Nick Manzella. This version of the code implements weighting in the y and xi directions, masking in the y and xi directions, an arbitrary group velocity feature, and the ability to create 3D probes with size limitations at larger numbers of particles due to memory issues. 

The `version-1.0` branch contains the first complete version of the code, as left by Marisa Petrusky. No output files are saved in this version; plots are immediately generated. Multiprocessing is also not implemented here.

The respective `unit-test` branches contain input files and plotting scripts used to verify QuEP. See Marisa's [UG thesis](https://www.researchgate.net/publication/351853356_Picturing_Plasma_Studying_the_Simulated_Transverse_Probing_of_Laser_Wakefield_Accelerators) for walkthroughs of her verification testing. Running the `unitTest.py` *within these branches*, then cross-checking the numbers with Marisa's thesis can be used to verify if QuEP is working correctly on your computer.

The `plotting` branch is used as a working branch.

### Library Structure 

`main.py` reads the initialization file, generates a probe with the `shape.initProbe` function, then sends the conditions to `eProbe.py`. 

`shape.initProbe` refers to a subprogram from `/include/shapes/`, where each file creates an array of electrons in the desired shape. To add new shapes, create a file within the shapes directory, then add the initialization condition to `main.py` (~Line 85). 

`eProbe.py` propagates electron motion inside electromagnetic fields using a first order Runge-Kutta method. It finds the electromagnetic force at any given point using the `sim.EField` and `sim.BForce` functions (~Line 99). 

#### Changing Pre-Generated Simulation Files

`sim.EField` and `sim.BForce` refers to a subprogram from `/include/simulations/`, where each file contains *unique* functions for reading out data from pre-generated electromagnetic field files; plasma density, frequency, and time values; boundary conditions of the plasma cell; and the `EField`, `BForce`, and `BField` functions. 

To add a new set of simulation data, create a file within the simulations directory, then add the initalization condition to `main.py` (near Line 74), as well as any plotting scripts in use. The new file absolutely MUST include the functions `getTime`, `getBoundCond`, `EField`, and `BForce`. Other functions, such as `getPlasDensity` and `getPlasFreq`, are used in plotting results, but not running QuEP itself. 

### Comments

* QuEP is not optimized algorithm-wise, so even with the Python multiprocessing package, high density probes (e.g. >1e6 particles ) will take several hours to run on personal computers. 

* Because this is a numerical simulation, there is no exact way to verify if your output probe is "correct"! We can only decide whether what we see aligns with our physics intuition. 

* As noted before, there are limitations to the 3D probes feature as you will quickly run out of memory on both personal and supercomputers with high density probes.

### Requirements
This simulation uses Python 3.0, and requires the packages `h5py`, `importlib`, `numpy`, and `multiprocessing`. Plots require `matplotlib`.

### Contact
Contact Evan Trommer (evtrommer[at]gmail.com), Nicholas Manzella (nick.manzella31[at]gmail.com) or Marisa Petrusky (marisapetrusky[at]gmail.com) for questions about this code. Source code can be found at https://github.com/SBU-PAG/QuEP/

#### Theses
For more information on this project, you can read our senior thesises here:

Evan Trommer (Stony Brook University, 2023): *To be uploaded soon.*

Nick Manzella (Stony Brook University, 2022): [Development of methods for modeling the interactions of plasma wakefields with a realistic 3D electron probe](https://1drv.ms/b/s!AkeL_dqkZf-PieYi7_ddYZSPNQklPg?e=ayKaUf)

Marisa Petrusky (Stony Brook University, 2021): [Picturing Plasma: Studying the Simulated Transverse Probing of Laser Wakefield Accelerators](https://www.researchgate.net/publication/351853356_Picturing_Plasma_Studying_the_Simulated_Transverse_Probing_of_Laser_Wakefield_Accelerators)

Audrey Farrell (Stony Brook University, 2020): Simulating beam induced ionization-injectionin plasma wakefield accelerators (*Email Evan or Marisa for copy*)
