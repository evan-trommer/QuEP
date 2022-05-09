# Here is where the initial conditions of the electron probe are defined
# This filename is the input parameter of the eProbe.py file

# This file will be used to test the group velocity feature of the code with
# a single electron through the blowout regime. 

simulation_name = 'QUASI3D'
shape = 'single'
iterations = 100000
mode = 0
fname = "group_velocity_tests_002.npz"
debugmode = True

# Probe centered at the following initial coordinates (in c/w_p):
x_c = -2.0 # Start within region of field
y_c = 1.5
xi_c = -7.3 #Makes Z = about 45,  since t0= 51.9948 for 000130

# Initial momentum
px_0 = 100 # Make sure it goes towards the screen!
py_0 = 0
pz_0 = 1000

# Screen Parameters (Assume infinite in y and z)
x_s = 500

# Shape Parameters (Radius or Side Length)
s1 = None # In y
s2 = None # In xi

# Densities
ydensity = 1
xidensity = 1
xdensity = 1 # Probe width - single layer
resolution = None