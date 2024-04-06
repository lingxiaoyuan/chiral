import os,sys
sys.path.append("/projectnb/lejlab2/lxyuan/ABAQUSF/chiral/scripts/")
sys.path.append('/projectnb/twodtransport/lxyuan/chiral/tools')
import math
import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from FEmodel_arm3 import *
from lig_space import *

ks = {'mesh_size':0.02,  #smallest mesh size
    'mesh_bias':20,   # largerst mesh size = bias*smallest mesh size
    'data_points':40, # data points for each step
    'geoNL': False,  # geometry nonlinearity 
    'E': 70.0,  #important: 70000 -> 70
    'poisson': 0.3,
    'step_names': ['Initial', 'Step-1'],
    'ds':mix_space(['poly']*10),   #design space
    'sym':False}
keys = ['a','R', 'L','theta0', 'theta1','theta00','theta11','r_s']  

for path in range(int(sys.argv[-2]),int(sys.argv[-1])):
    try:
        a= np.loadtxt('a%s.dat'%path)
        circle = np.loadtxt('circle%s.dat'%path)
        values = [list(a)]+ list(circle)

        for key,value in zip(keys,values):
            ks[key] = value
        
        for i in range(4):
            #dilation -> rotation
            try:
                u1,u2 = np.loadtxt('disp%s_test%s.dat'%(path,i))
            except:
                continue
                #u1,u2 = np.array((0.1,0.1))*(2*i-1)
            u = ((u1,u2),)
            name = 'a%s_one_ligament_test%s'%(path,i)
            simulation(u = u, name = name, **ks)
    except:
        continue