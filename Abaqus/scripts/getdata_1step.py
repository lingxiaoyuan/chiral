# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:20:38 2022

@author: Yuan
"""



#!/user/bin/python
#-*-coding:UTF-8-*-
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import visualization
import displayGroupOdbToolset as dgo 
import odbAccess
import math
import os.path
import numpy as np
from collections import defaultdict


def datacollection(filename):
    
    oldpath = filename+'.odb'
    
    if not os.path.exists(oldpath):
        print('No such file') 
        return
    #newpath = './m14/'+name+'/new_outputf.odb'
    #session.upgradeOdb(existingOdbPath=oldpath, upgradedOdbPath=newpath)
    odb=openOdb(path = oldpath)
    
    FUdata = []
    
    for step in ['Step-1', ]:
        Move=odb.steps[step]
        for name_index in range(len(Move.historyRegions.keys())):
                ##right circle
                if 'CIRCLE-1-LIN-2-1.' in Move.historyRegions.keys()[name_index]:
                    nodename = Move.historyRegions.keys()[name_index]
        
        for key in ['U1', 'U2', 'UR3','RF1', 'RF2', 'RM3']:
            Values = Move.historyRegions[nodename].historyOutputs[key].data
            FUdata.append(np.array(Values)[:,1])
            
    for step in ['Step-1',]:
        Move=odb.steps[step]
        Values = Move.historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLSE'].data
        FUdata.append(np.array(Values)[:,1])

    ##output sum of contact pressure
    CPdata = []
    for step in ['Step-1',]:
        Move=odb.steps[step]
        for frame in Move.frames:
            cpress=frame.fieldOutputs['CPRESS'].values
            cpressValues=[cpress[i].data for i in range(len(cpress))]
            CPdata.append(sum(cpressValues))
        FUdata.append(np.array(CPdata))
        
    np.savetxt(filename+'_output.dat',np.array(FUdata).T,fmt='%.6f')
    np.savetxt(filename+'_outputcp.dat',np.array(CPdata).T,fmt='%.6f')

    
    ##save the points that are in contact
    frame = Move.frames[-1]
    contactPoints = np.ones((0,5))
    cpress=frame.fieldOutputs['CPRESS'].values
    U = frame.fieldOutputs['U']
    for item in cpress:
        if item.instance.name=='LIGAMENT-1':
            cpressvalue = item.data 
            nodelabel = item.nodeLabel    
            coords = item.instance.nodes[nodelabel-1].coordinates       #node was labeled from '1' while the numpy array index starts from '0'
            #coords of deformed ligament
            Xdis=U.getSubset(region=item.instance.nodes[nodelabel-1]).values[0].data[0]
            Ydis=U.getSubset(region=item.instance.nodes[nodelabel-1]).values[0].data[1]
            row = [coords[0],coords[1],cpressvalue,Xdis,Ydis]
            contactPoints = np.row_stack([contactPoints,row])

    np.savetxt(filename+'_lastframe_cpoints.dat',contactPoints,fmt='%.6f')      

    #get strain and stress
    #columns = ['Elabel', 'Nlabel','X','Y','SF','SM','EV','ELSE]
    ssv = []
    EV = frame.fieldOutputs['EVOL']  #element volumn 
    SF = frame.fieldOutputs['SF'] 
    SM = frame.fieldOutputs['SM'] 
    ELSE = frame.fieldOutputs['ELSE'] 

    coords_elements = defaultdict(dict)
    SF_elements = defaultdict(dict)
    SM_elements = defaultdict(dict)
    EV_elements = {}
    ELSE_elements = {}

    #every element has two ending points(two nodeLabel) and the values are close so only one was recoreded and was used to represent the values of the whole element
    for i in range(len(SF.values)):
        assert SF.values[i].instance.name =='LIGAMENT-1'
        elementlabel = SF.values[i].elementLabel
        nodelabel = SF.values[i].nodeLabel 
        assert SM.values[i].elementLabel == elementlabel
        assert SM.values[i].nodeLabel  == nodelabel
        SF_elements[elementlabel][nodelabel] = SF.values[i].data[0]
        SM_elements[elementlabel][nodelabel] = SM.values[i].data[0]
        assert SF.values[i].instance.nodes[nodelabel-1].label == nodelabel
        coords_elements[elementlabel][nodelabel] = SM.values[i].instance.nodes[nodelabel-1].coordinates 

    for i in range(len(EV.values)): 
        elementlabel = EV.values[i].elementLabel    
        assert EV.values[i].instance.name == 'LIGAMENT-1'
        assert ELSE.values[i].elementLabel == elementlabel
        EV_elements[elementlabel] = EV.values[i].data 
        ELSE_elements[elementlabel] = ELSE.values[i].data 
        for nodelabel in coords_elements[elementlabel].keys():
            row = [elementlabel,
                    nodelabel,
                    coords_elements[elementlabel][nodelabel][0],
                    coords_elements[elementlabel][nodelabel][1],
                    SF_elements[elementlabel][nodelabel],
                    SM_elements[elementlabel][nodelabel],
                    EV_elements[elementlabel],
                    ELSE_elements[elementlabel],
                    ]
            ssv.append(row)
        #ssv = np.row_stack([ssv,row])
    np.savetxt(filename+'_lastframe_ssv.dat',np.array(ssv),fmt='%.6f')

if __name__ == "__main__":
    for path in range(int(sys.argv[-2]),int(sys.argv[-1])):
        for filename in ['/one_ligament_test%s'%i for i in range(4)]:
            try:
                datacollection(str(path)+filename)  
            except:
                continue




