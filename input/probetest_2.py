#Input file for a probe travelling through the blowout regime in simulation '000130'
#Created on: 6/13/2022

simulation_name = 'QUASI3D'
shape = 'rprism'
# dt = 0.005, 150000
iterations = 500000
mode = 0 #0 is just wake fields, 1 is laser fields, -1 for both
fname = "wakefield_test.npz"
debugmode = False

# Probe centered at the following initial coordinates (in c/w_p):
x_c = -1. # Start within region of field # 2.4 = maximum x_c
y_c = 0.
xi_c = -9.5

# Initial momentum
px_0 = 110 # Make sure it goes towards the screen!
py_0 = 0
pz_0 = 0

# Screen Distances (from z-axis of plasma cell, in mm):
x_s = [10, 50, 100, 250, 500]

# Shape Parameters (Radius or Side Length, in c/w_p):
s1 = 1 # In y
s2 = 5 # In xi

# Densities
ydensity = 500
xidensity = 2500
xdensity = 1 # Probe width - single layer
resolution = 0.002
