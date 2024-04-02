import math
import numpy as np
import os,sys
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
sys.path.append("/projectnb/twodtransport/lxyuan/ABAQUSF/chiral/scripts/")
from FEmodel import *

ks = {'mesh_size':0.01,  #smallest mesh size
    'mesh_bias':20,   # largerst mesh size = bias*smallest mesh size
    'data_points':50, # data points for each step
    'geoNL': False,  # geometry nonlinearity 
    'E': 70000.0,
    'poisson': 0.3}
sym = False
#lig1: No.19 in file asym_circle_8_8_100.dat
a =  [-1.608844886853757483e-01, 5.169784022521111133e-01, 1.266116907473802122e-02, -1.018990904567020905e-01, 7.532510118309368452e-03, 6.694802703993044113e-03, -2.874426646035281932e-04, -1.466138275192054225e-04]
circle = [6.786858283004024273e+00, 2.000000000000000000e+01, -2.637779611909901689e+01, -5.331068656739279277e+01, 1.538888615880822499e+02, -7.227603942694929628e+01]

#lig2: No.8 in file asym_circle_8_8_100.dat
#a =  [5.321557691783483923e-01, 1.850746773994059180e-01, -2.004079709466848047e-02, -7.281104461151080912e-03, 4.353951090651722458e-04, 9.432696339012049081e-05, -2.700810294921138070e-06, -4.007764465959412977e-07]
#circle = [3.260241804571311963e+00, 2.000000000000000000e+01, -5.077807743859078471e+01, -8.053953078020539635e+01, 1.164440723558185482e+02, -7.275995471048661045e+01]
###key argumenst:geometry
keys = ['a','R', 'L','theta0', 'theta1','theta00','theta11']  
values = [a]+circle
for key,value in zip(keys,values):
    ks[key] = value

meshsize = [0.01,0.02,0.05,0.1,0.5,1.0,0.03,0.04]
u1s = [0.05,0.2,0.5,1.0,0.1]
u2s = [0.05,0.2,0.5,1.0,0.1]

for geo,geoname  in zip([False,True],['geoNLoff','geoNLon']):
    ks['geoNL'] = geo
    for i in range(6,len(meshsize)):
        ks['mesh_size'] = meshsize[i]
        for j in range(len(u1s)):
            #dilation -> rotatio1a
            u1,u2 = u1s[j], u2s[j]
            u = ((u1,0),(u1,u2),(0,u2),(0,0))
            simulation(u = u, sym = sym, name = '%s_mesh%s_disp%s'%(geoname,i,j), **ks)

