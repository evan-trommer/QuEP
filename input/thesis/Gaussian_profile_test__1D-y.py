# Here is where the initial conditions of the electron probe are defined
# This filename is the input parameter of the eProbe.py file
# Used to create single line of particles in x-direction to test gaussian weighting

simulation_name = 'QUASI3D'
shape = 'rprism_weighted_after'
# dt = 0.005, 150000
iterations = 500000
mode = -1
fname = "Gaussian_profile_test__1D-y.npz"
debugmode= False

# Probe centered at the following initial coordinates (in c/w_p):
x_c = 2.4 # Start at ending position
y_c = 0
xi_c = -13

# Initial momentum
px_0 = 110 # Make sure it goes towards the screen!
py_0 = 0
pz_0 = 0

# Screen Distances (from z-axis of plasma cell, in mm):
#x_s = [100, 200, 300, 400, 500] # Only 5
x_s = [10, 50, 100, 250, 500]

# Shape Parameters (Radius or Side Length, in c/w_p):
s1 = 1 # In y
s2 = 5 # In xi

# Densities
ydensity = 100
xidensity = 1
xdensity =  1 #This effectively creates the number of layers in x direction
resolution = 0.002

