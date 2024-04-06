import math
import numpy as np
import os,sys
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)

sys.path.append('/projectnb/twodtransport/lxyuan/chiral/chiral_analysis')
from lig_space import *


################################################ Part #####################################################
def part_circle(R,L):
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(-L/2, 0.0), point1=(-L/2,R))
    p = mdb.models['Model-1'].Part(name='circle', dimensionality=TWO_D_PLANAR, 
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['circle']
    p.BaseWire(sketch=s)
    s.unsetPrimaryObject()
    #session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    p = mdb.models['Model-1'].parts['circle']
    v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[0], rule=CENTER))

    p = mdb.models['Model-1'].parts['circle']
    edges1 = p.edges.findAt(((-L/2,-R,0.0),),((-L/2,R,0.0),))
    p.Set(edges=edges1, name='circle')

################################################ Material Property #####################################################
def material_property(E = 70000.0, poisson = 0.3):
    mdb.models['Model-1'].Material(name='Material-1')
    mdb.models['Model-1'].materials['Material-1'].Elastic(table=((E, poisson), ))
    mdb.models['Model-1'].materials['Material-1'].Density(table=((1.0, ), ))

    mdb.models['Model-1'].RectangularProfile(name='Profile-1', a=30.0, b=1.5)
    mdb.models['Model-1'].BeamSection(name='Section-1', 
        integration=DURING_ANALYSIS, poissonRatio=0.0, profile='Profile-1', 
        material='Material-1', temperatureVar=LINEAR, consistentMassMatrix=False)
    
    p = mdb.models['Model-1'].parts['ligament']
    region=p.sets['ligament']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    #: Beam orientations have been assigned to the selected regions.
    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))

################################################### Assembly ############################################################## 
def assembly(L):
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['circle']
    a.Instance(name='circle-1', part=p, dependent=ON)
    a = mdb.models['Model-1'].rootAssembly
    a.LinearInstancePattern(instanceList=('circle-1', ), direction1=(1.0, 0.0, 0.0), direction2=(0.0, 1.0, 0.0), number1=2, number2=1, spacing1=L,  spacing2=1.0)

    a = mdb.models['Model-1'].rootAssembly
    a.rotate(instanceList=('circle-1', ), axisPoint=(-L/2, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=270.0)
    #a.rotate(instanceList=('circle-1-lin-2-1', ), axisPoint=(L/2, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=90.0)
        
    a = mdb.models['Model-1'].rootAssembly
    p = mdb.models['Model-1'].parts['ligament']
    a.Instance(name='ligament-1', part=p, dependent=ON)

################################################### Step ############################################################## 
def step(data_points = 40, geoNL = False, step_names = ['Initial', 'Step-1', 'Step-2','Step-3', 'Step-4']):
    for i in range(len(step_names)-1):
        mdb.models['Model-1'].StaticStep(name=step_names[i+1], previous=step_names[i], 
            initialInc=1.0/data_points)
    if geoNL:
        mdb.models['Model-1'].steps['Step-1'].setValues(nlgeom=ON)

    #set field output
    mdb.models['Model-1'].FieldOutputRequest(name='F-Output-1', 
        createStepName='Step-1', position=NODES, variables=('S', 'E', 
        'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 'EVOL','NFORCSO','ELSE','SF'))

    #set history output
    mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-1', 
    createStepName='Step-1', variables=PRESELECT, numIntervals=data_points)

    regionDef=mdb.models['Model-1'].rootAssembly.sets['leftCircle']
    mdb.models['Model-1'].HistoryOutputRequest(name='H-leftCircle', 
        createStepName='Step-1', variables=('U1', 'U2', 'UR3', 'RF1', 'RF2', 'RM3', 
        'CSTRESS', 'CFNM', 'CFN1', 'CFN2', 'CFN3', 'CAREA'), numIntervals=data_points, region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    regionDef=mdb.models['Model-1'].rootAssembly.sets['rightCircle']
    mdb.models['Model-1'].HistoryOutputRequest(name='H-rightCircle', 
        createStepName='Step-1', variables=('U1', 'U2', 'UR3', 'RF1', 'RF2', 'RM3',
        'CSTRESS', 'CFNM', 'CFN1', 'CFN2', 'CFN3', 'CAREA'), numIntervals=data_points, region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

    regionDef=mdb.models['Model-1'].rootAssembly.sets['ligament_leftEnd']
    mdb.models['Model-1'].HistoryOutputRequest(name='H-leftEnd', 
        createStepName='Step-1', variables=('U1', 'U2', 'UR3', 'RF1', 'RF2', 'RM3'), numIntervals=data_points, region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    regionDef=mdb.models['Model-1'].rootAssembly.sets['ligament_rightEnd']
    mdb.models['Model-1'].HistoryOutputRequest(name='H-rightEnd', 
        createStepName='Step-1', variables=('U1', 'U2', 'UR3', 'RF1', 'RF2', 'RM3'), numIntervals=data_points, region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

    

##################################################### Boundary condition #########################################  
# dimension of u: N*2  (N steps and for each step, clarify (ux,uy)), len(u) = len(step_names)-1
def boundary_condition(u, step_names = ['Initial', 'Step-1', 'Step-2','Step-3', 'Step-4']):
    a = mdb.models['Model-1'].rootAssembly
    region = a.sets['leftCircle']
    mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
        region=region, u1=SET, u2=SET, ur3=SET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)

    a = mdb.models['Model-1'].rootAssembly
    region = a.sets['rightCircle']
    mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
        region=region, u1=SET, u2=SET, ur3=SET, amplitude=UNSET, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    for i, name in enumerate(step_names[1:]):
        mdb.models['Model-1'].boundaryConditions['BC-2'].setValuesInStep(
            stepName=name, u1=u[i][0],u2=u[i][1])
############################################################## Contact ####################################################
def contact():
    a = mdb.models['Model-1'].rootAssembly
    #contact of left circle and ligament
    mdb.models['Model-1'].ContactProperty('IntProp-1')
    mdb.models['Model-1'].interactionProperties['IntProp-1'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, contactStiffness=DEFAULT, 
        contactStiffnessScaleFactor=1.0, clearanceAtZeroContactPressure=0.0, 
        constraintEnforcementMethod=AUGMENTED_LAGRANGE)
    #: The interaction property "IntProp-1" has been created.


    region1 = a.surfaces['circle_left']
    region2 = a.surfaces['ligament_left']
    mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='Int-1', 
        createStepName='Initial', master=region1, slave=region2, sliding=FINITE, 
        enforcement=NODE_TO_SURFACE, thickness=OFF, 
        interactionProperty='IntProp-1', surfaceSmoothing=NONE, adjustMethod=NONE, 
        smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

    region1 = a.surfaces['circle_right']
    region2 = a.surfaces['ligament_right']
    mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='Int-2', 
        createStepName='Initial', master=region1, slave=region2, sliding=FINITE, 
        enforcement=NODE_TO_SURFACE, thickness=OFF, 
        interactionProperty='IntProp-1', surfaceSmoothing=NONE, adjustMethod=NONE, 
        smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)


    #: The interaction "Int-2" has been created.
    #tie of left circle and ligament
    region1 = a.surfaces['circle_left']
    region2 = a.sets['ligament_leftEnd']
    mdb.models['Model-1'].Tie(name='Constraint-1', master=region1, slave=region2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)

    #tie of right circle and ligament
    region1 = a.surfaces['circle_right']
    region2 = a.sets['ligament_rightEnd']
    mdb.models['Model-1'].Tie(name='Constraint-2', master=region1, slave=region2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)   

############################################################## Mesh ####################################################
def meshing(mesh_size,mesh_bias):

    p = mdb.models['Model-1'].parts['ligament']
    e = p.edges
    #mesh size on the ligament that contact with the circle is min_mesh_size
    pickedEdges = p.sets['ligament'].edges
    p.seedEdgeBySize(edges=pickedEdges, size=mesh_size, deviationFactor=0.1, constraint=FINER)
    #modify the mesh size on the ligament that detachs from the circle to be biased mesh 
    pickedEndEdges = p.sets['ligament_middle'].edges
    p.seedEdgeByBias(biasMethod=DOUBLE, endEdges=pickedEndEdges, minSize=mesh_size, maxSize=mesh_size*mesh_bias, constraint=FINER)
    p.generateMesh()

    elemType1 = mesh.ElemType(elemCode=B23, elemLibrary=STANDARD)
    pickedRegions =(pickedEdges, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))


    p = mdb.models['Model-1'].parts['circle']
    pickedEdges = p.sets['circle'].edges
    #p.seedEdgeBySize(edges=pickedEdges, size=mesh_size, deviationFactor=0.1, constraint=FINER)
    p.seedEdgeByBias(biasMethod=DOUBLE, endEdges=pickedEdges, minSize=mesh_size,  maxSize=mesh_size*mesh_bias, constraint=FINER)

    p.generateMesh()
    elemType1 = mesh.ElemType(elemCode=R2D2, elemLibrary=STANDARD)
    pickedRegions =(pickedEdges, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))

    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()

############################################################## Write output file ####################################################
def write_input(file_name):
    mdb.Job(name=file_name, model='Model-1', description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
                numGPUs=0)
    mdb.jobs[file_name].writeInput(consistencyChecking=OFF)

############################################################## Simulation ####################################################
def simulation(u,name = 'temp',**kwargs):
    Mdb()
    sym = kwargs['sym']
    a = kwargs['a']
    ds = kwargs['ds']
    R, L,theta0,theta1,r_s = kwargs['R'] ,kwargs['L'], kwargs['theta0'], kwargs['theta1'], kwargs['r_s']    
    theta0 = math.radians(theta0)
    theta1 = math.radians(theta1)
    start = theta0
    end = theta0+theta1
    x0,y0 = -L/2+R*math.cos(start), R*math.sin(start)
    x1,y1 = -L/2+R*math.cos(end), R*math.sin(end)

    if sym:
        x00,y00 = -x0, -y0
        x11,y11 = -x1, -y1
        theta11 = theta1
    else:
        theta00, theta11 = kwargs['theta00'], kwargs['theta11']
        theta00 = math.radians(theta00)
        theta11 = math.radians(theta11)
        start = theta00
        end = theta00+theta11
        x00,y00 = L/2+R*math.cos(start), R*math.sin(start)
        x11,y11 = L/2+R*math.cos(end), R*math.sin(end)
        #xmidd,ymidd = L/2+R*math.cos(start/2+end/2), R*math.sin(start/2+end/2)

    x1s = -L/2+ (R+2*r_s)*np.cos(theta0)
    y1s = (R+2*r_s)*np.sin(theta0)
    ############################################################## Parts - Circle ###################################################
    part_circle(R,L)

    ############################################################## Parts - Ligament ###################################################
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)

    #ligament part that overlap with the circle
    if theta1>0:
        s1.ArcByCenterEnds(center=(-L/2, 0.0), point1=(x1, y1), point2=(x0, y0), direction=CLOCKWISE)
    else:
        s1.ArcByCenterEnds(center=(-L/2, 0.0), point1=(x1, y1), point2=(x0, y0), direction=COUNTERCLOCKWISE)

    if theta11>0:
        s1.ArcByCenterEnds(center=(L/2, 0.0), point1=(x00, y00), point2=(x11, y11), direction=COUNTERCLOCKWISE)
    else:
        s1.ArcByCenterEnds(center=(L/2, 0.0), point1=(x00, y00), point2=(x11, y11), direction=CLOCKWISE)

    #ligament part that does not overlap with the circle
    xleft = list(np.linspace(x1s,0,250)) 
    yleft = [sumf(c=a,f = ds.f, x=i) for i in xleft]
    #yleft = [polynomial(n = len(a), c = a, x = i) for i in xleft]
    if sym:
        xright = [-i for i in xleft[::-1]]
        yright = [-i for i in yleft[::-1]]
    else:
        xright = list(np.linspace(0,x00,250))
        yright = [sumf(c=a,f = ds.f, x=i) for i in xright]
        #yright = [polynomial(n = len(a), c = a, x = i) for i in xright]
    x = xleft+xright
    y = yleft+yright
    s1.Spline(points=tuple((x[i],y[i]) for i in range(len(x))))

    #ligament on the small circle

    x_central = -L/2+ (R+r_s)*np.cos(theta0)
    y_central = (R+r_s)*np.sin(theta0)
    s1.ArcByCenterEnds(center=(x_central, y_central), point1=(x0, y0), point2=(x1s, y1s), direction=COUNTERCLOCKWISE)
    
    #creat part "ligament"   
    p = mdb.models['Model-1'].Part(name='ligament', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['ligament']
    p.BaseWire(sketch=s1)
    s1.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']

    ######################################################### Part Sets for Assign Material Property and Meshing ##########################################################
    p = mdb.models['Model-1'].parts['ligament']
    #ligament part contact with left circle
    edge1 = p.edges.findAt(((x1,y1,0.0),))
    #middle part
    edge2 = p.edges.findAt(((x[2],y[2],0.0),))
    #ligament part contact with right circle
    edge3 = p.edges.findAt(((x11,y11,0.0),))
    
    #ligament part - small circle
    edge4 = p.edges.findAt(((x0,y0,0.0),))
    
    p.Set(edges=edge1+edge2+edge3+edge4, name='ligament')
    p.Set(edges=edge2, name='ligament_middle')

    ######################################################### Assembly ##########################################################
    assembly(L)

    ######################################################### Assembly Sets/Surfaces for contact setting ##########################################################
    a = mdb.models['Model-1'].rootAssembly
    #ligament part contact with left circle
    edge1 = a.instances['ligament-1'].edges.findAt(((x1,y1,0.0),))
    #middle part
    edge2 = a.instances['ligament-1'].edges.findAt(((x[2],y[2],0.0),))
    #ligament part contact with right circle
    edge3 = a.instances['ligament-1'].edges.findAt(((x11,y11,0.0),))

    #ligament part - small circle
    edge4 = a.instances['ligament-1'].edges.findAt(((x0,y0,0.0),))
    
    a.Set(edges=edge1+edge2+edge3+edge4, name='ligament')

    # surfaces for contact
    a.Surface(side2Edges=edge1+edge4, name='ligament_left')
    a.Surface(side2Edges=edge2+edge3, name='ligament_right')

    edges1 = a.instances['circle-1'].edges.findAt(((-L/2,R,0.0),))
    a.Surface(side2Edges=edges1, name='circle_left')
    edges1 = a.instances['circle-1-lin-2-1'].edges.findAt(((L/2,R,0.0),))
    a.Surface(side2Edges=edges1, name='circle_right')

    #sets for tie
    verts1 = a.instances['ligament-1'].vertices.findAt(((x1,y1,0),),)
    verts2 = a.instances['ligament-1'].vertices.findAt(((x11,y11,0),))
    a.Set(vertices=verts1, name='ligament_leftEnd')
    a.Set(vertices=verts2, name='ligament_rightEnd')

    #sets for history output
    #r1 = {2: 'ReferencePoint object'}
    r1 = a.instances['circle-1'].referencePoints
    a.Set(referencePoints=(r1[2], ), name='leftCircle')
    #: The set 'leftCircle' has been created (1 reference point).
    r1 = a.instances['circle-1-lin-2-1'].referencePoints
    a.Set(referencePoints=(r1[2], ), name='rightCircle')
    #: The set 'rightCircle' has been created (1 reference point).
    
    ################################################ Material Property ##############################################
    material_property(kwargs['E'], kwargs['poisson'] )
    ###################################################### Step #####################################################
    step(kwargs['data_points'],kwargs['geoNL'],kwargs['step_names'])
    ##################################################### Boundary condition #########################################  
    boundary_condition(u,kwargs['step_names'])
    ##################################################### Contact #########################################  
    contact()
    ######################################################## Mesh ############################################################### 
    meshing(kwargs['mesh_size'],kwargs['mesh_bias'])
    ######################################################## Input files############################################################### 
    write_input(name)


