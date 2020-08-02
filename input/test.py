# Here is where the initial conditions of the electron probe are defined
# This filename is the input parameter of the eProbe.py file

simulation_name = 'QUASI3D'
shape = 'single'
density = 1
iterations = 15000
mode = -1

# Probe centered at the following initial coordinates (in c/w_p):
x_c = 0.15 # Start within region of field
y_c = 0
xi_c = -11

# Initial momentum
px_0 = 20 # Make sure it goes towards the screen!
py_0 = 0
pz_0 = 0

# Screen Distances (from z-axis of plasma cell, in mm):
#x_s = [100, 200, 300, 400, 500] # Only 5
x_s = [10, 50, 100, 250, 500]

# Shape Parameters (Radius or Side Length, in c/w_p):
s1 = 2#0.5 # In y
s2 = 10 # In xi
s3 = 1 # In x
