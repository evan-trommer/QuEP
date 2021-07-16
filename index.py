# Include file imports
import sys
import importlib
import numpy as np
import include.plot2DTracks as plot2D
import include.plot3DTracks as plot3D
import include.showQuickEvolution as showEvol_Q
import include.showFullEvolution as showEvol_F
import include.viewProbe as viewProbe
import include.writeFullEvolData as writeHist
import include.shapes.postmasks_y as postmasks_y
import include.shapes.postmasks_xi as postmasks_xi
# Be sure to change file name location!

# Plotting Scripts
plot2DTracks = False                 # View 2D projections of trajectories
showQuickEvolution = False           # View evolution of probe after leaving plasma at inputted x_s in scatter plots
showFullEvolution = False          # View full evolution of probe at hardcoded locations in colored histograms
writeHistData = False
# Set all others equal False if want animation saved (dependency issue)
saveMovie = False                   # Save gif of probe evolution
if (saveMovie):
    import include.makeAnimation as makeAnimation

input_fname = str(sys.argv[1])
print("Using initial conditions from ", input_fname)
init = importlib.import_module(input_fname)
sim_name = init.simulation_name
shape_name = init.shape
yden = init.ydensity
xiden = init.xidensity
res = init.resolution
iter = init.iterations
mode = init.mode
fname = init.fname
x_c = init.x_c
y_c = init.y_c
xi_c = init.xi_c
px_0 = init.px_0
py_0 = init.py_0
pz_0 = init.pz_0
x_s = init.x_s
s1 = init.s1
s2 = init.s2
s3 = init.s3

#Define masks in y direction, 0 is 0 on the y-axis. Change if different mask is desired
top_of_masks = []  #upper limit of each mask in order
bot_of_masks = []  #lower limit of each mask in order 

#Define masks in z direction, leftmost z-coordinate = 0. Change if different mask is desired
left_of_masks= []  #left most limit of each mosk in order
right_of_masks = []  #right most limit of each mask in order

data = np.load('./data/' + fname) # Change this line as needed
x_f = data['x_dat']
y_f = data['y_dat']
xi_f = data['xi_dat']
z_f = data['z_dat']
px_f = data['px_dat']
py_f = data['py_dat']
pz_f = data['pz_dat']
t0 = data['t_dat']

new_ydensity = yden  #just in case there are no horizontal masks

if len(top_of_masks)>0:
    x_f,y_f,xi_f,z_f,px_f,py_f,pz_f,new_ydensity = postmasks_y.initProbe(x_c,y_c,xi_c,t0,s1,s2,s3,yden,xiden,res,top_of_masks,bot_of_masks,x_f,y_f,xi_f,z_f,px_f,py_f,pz_f)
    
if len(left_of_masks)>0:
    x_f,y_f,xi_f,z_f,px_f,py_f,pz_f = postmasks_xi.initProbe(x_c,y_c,xi_c,t0,s1,s2,s3,yden,xiden,res,left_of_masks,right_of_masks,x_f,y_f,xi_f,z_f,px_f,py_f,pz_f,new_ydensity)

noElec = len(x_f)

# Plot data points
if (plot2DTracks):
    plot2D.plot(x_f, y_f, xi_f, z_f, px_f, py_f, pz_f, sim_name, shape_name, x_s, noElec)
if (showQuickEvolution):
    showEvol_Q.plot(x_f, y_f, xi_f, z_f, px_f, py_f, pz_f, sim_name, shape_name, x_s, noElec, iter)
if (showFullEvolution):
    showEvol_F.plot(x_f, y_f, xi_f, z_f, px_f, py_f, pz_f, sim_name, shape_name, noElec, iter)
if (writeHistData):
     writeHist.plot(x_f, y_f, xi_f, z_f, px_f, py_f, pz_f, sim_name, shape_name, noElec, iter)
if (saveMovie):
    makeAnimation.animate(x_f, y_f, xi_f, z_f, px_f, py_f, pz_f, sim_name, shape_name, noElec, iter)