# Here is where the initial conditions of the electron probe are defined
# This filename is the input parameter of the eProbe.py file
#
# This input file is for testing the field data for a0 = 4 sim (sim ID 000067)
# Wide sweep of 000067

simulation_name = 'QUASI3D'
shape = 'rprism_weighted_after'
iterations = 100000
mode = 0
fname = "000034_testing.npz"
debugmode = False

# Probe centered at the following initial coordinates (in c/w_p):
x_c = -8 # Start within boundary of sim, outside bubble, on correct side
y_c = 0 # Off-center to avoid center of bubble?
xi_c = -15

# Initial momentum
px_0 = 100 # Make sure it goes towards the screen!
py_0 = 0
pz_0 = 0

# Screen Parameters (Assume infinite in y and z)
x_s = [10, 50, 100, 250, 500]

# Shape Parameters (Radius or Side Length)
s1 = 5 # In y
s2 = 10 # In xi

# Densities
ydensity = 750
xidensity = 1500
xdensity = 1 # Probe width - 1 single layer
resolution = None