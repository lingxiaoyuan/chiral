import numpy as np
import pandas as pd
import math
import random
import os,sys
import pickle
from pathlib import Path
from numpy.linalg import lstsq,norm 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
plt.rcParams['font.size'] = 20

from tools.helpers import *
#from tools.ligament_design import *
from tools.lig_space import *

#plot the geometry
from tools.arm0_design import show_geo_abaqus as arm0_show_geo_abaqus
from tools.arm1_design import show_geo_abaqus as arm1_show_geo_abaqus
from tools.arm2_design import show_geo_abaqus as arm2_show_geo_abaqus
from tools.arm3_design import show_geo_abaqus as arm3_show_geo_abaqus

from tools.arm0_design import asym_poly_designs as arm0_asym_poly_designs
from tools.arm1_design import asym_poly_designs as arm1_asym_poly_designs
from tools.arm2_design import asym_poly_designs as arm2_asym_poly_designs
from tools.arm3_design import asym_poly_designs as arm3_asym_poly_designs
from tools.arm3_tools_aug import *


from tools.arm0_design import one_ligament_asym as arm0_one_ligament_asym
from tools.arm1_design import one_ligament_asym as arm1_one_ligament_asym
from tools.arm2_design import one_ligament_asym as arm2_one_ligament_asym
from tools.arm3_design import one_ligament_asym as arm3_one_ligament_asym


class compareStiff():
    def __init__(self):
          pass
    def compare_ij(self,s1,s2,i,j,i1=None,j1=None):
        if i1 and j1:
            return s1[i][j]/s2[i1][j1]     
        return s1[i][j]/s2[i][j]     

    def compare_r1(self,s1,s2):
        return (s1[0][0]-s2[0][0])/(s1[0][1]-s2[0][1])

    def compare_r2(self,s1,s2):
        return (s1[1][0]-s2[1][0])/(s1[1][1]-s2[1][1])

def error(y_target, y_pred):
    #return np.square(y_target-y_pred).sum()
    y_target = y_target.ravel()
    y_pred = y_pred.ravel()
    #cosine similarity
    return 1-np.dot(y_target,y_pred)/(norm(y_target,2)*norm(y_pred,2))

def get_r1r2(stiff):
    dxx = stiff[0] - stiff[0+4]
    dxy = stiff[1] - stiff[1+4]
    dyx = stiff[2] - stiff[2+4]
    dyy = stiff[3] - stiff[3+4]
    r1 = dxx/dxy
    r2 = dyx/dyy
    return r1,r2

def getArea(fs,us):
    n = int(len(fs)/4)
    du1 = abs(us[n,0]/n)
    du2 = abs(us[-n,1]/n)
    F1U1 = sum(np.abs(fs[:n, 0] - fs[2*n:3*n, 0][::-1]))*du1
    F2U2 = sum(np.abs(fs[n:2*n, 1] - fs[3*n:, 1][::-1]))*du2
    return F1U1,F2U2
    #abs(F1U1-F2U2),abs(F1U1-F2U2)/((F1U1+F2U2)/2.0)
   
def getScore(fs,us,stiff):
    #stiff is an array
    n = int(len(fs)/4)
    u1 = us[n,0]
    u2 = us[-n,1]
    r1,r2 = get_r1r2(stiff)
    
    ratio = (r1+r2)/2.0   
    
    if u1<0 and u2<0:
        K02 = stiff[[0,1,6,7]].reshape(2,2)
        K20 = stiff[[4,5,2,3]].reshape(2,2)
        #ratio = (r1+r2)/2.0   
    elif u1<0 and u2>0:
        K02 = stiff[[0,1,2,3]].reshape(2,2)
        K20 = stiff[[4,5,6,7]].reshape(2,2)
        #ratio = -(r1+r2)/2.0 
    elif u1>0 and u2<0:
        K02 = stiff[[4,5,6,7]].reshape(2,2)
        K20 = stiff[[0,1,2,3]].reshape(2,2) 
        #ratio = -(r1+r2)/2.0
    elif u1>0 and u2>0:
        K02 = stiff[[4,5,2,3]].reshape(2,2)
        K20 = stiff[[0,1,6,7]].reshape(2,2) 
        #ratio = (r1+r2)/2.0   

    intsec = np.argwhere(np.abs(us[:,1])>np.abs(us[:,0]*ratio))[0][0]
    fs_ref = np.row_stack([us[:intsec]@K02.T, us[intsec:]@K20.T]) 
    #e02 = error(fs[:intsec], us[:intsec]@K02.T)
    #e20 = error(fs[intsec:], us[intsec:]@K20.T) 
    e_ratio = abs(r1-r2)
    #return e02+e20+e_ratio*100
    #return max(e02,e20)
    e02 = error(fs[:n*2], us[:n*2]@K02.T)
    e20 = error(fs[n*2:], us[n*2:]@K20.T)  
    return [(e02+e20)*500, e_ratio]
    #return [error(fs, fs_ref)*1000, e_ratio]

def get_fs_us(data):
    us = data[:,:2]
    fs = data[:,3:5]
    return fs, us

def getStiff(path,index,threshold=float('inf')):
    sAll = []
    data0 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_output.dat'%0))
    data1 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_output.dat'%1))
    data2 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_output.dat'%2))
    data3 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_output.dat'%3))
    n=len(data0)
    
    for data in [data0,data1,data2,data3]:
        fs,us = get_fs_us(data)
        fs_temp,us_temp = fs[1:n], us[1:n]
        stiffM = lstsq(us_temp,fs_temp,rcond = None)[0]
        ape = abs((us_temp@stiffM-fs_temp)/fs_temp).mean()
        stiffM = stiffM.T
        if ape>threshold: 
            return np.zeros(8)
        else:
            sAll.append(stiffM)
    
    return np.array([sAll[0][0,0], sAll[1][0,1],sAll[0][1,0], sAll[1][1,1],sAll[2][0,0], sAll[3][0,1],sAll[2][1,0], sAll[3][1,1]])

def getContact(path,index):
    concact_list = np.ones(8)
    data0 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_outputcp.dat'%0))
    data1 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_outputcp.dat'%1))
    data2 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_outputcp.dat'%2))
    data3 = np.loadtxt(os.path.join(path, str(index), 'one_ligament_test%s_outputcp.dat'%3))
    # 0 represent no contact
    if (data0[1:-1]<1).all():
        concact_list[0] = 0
        concact_list[2] = 0
    if (data1[1:-1]<1).all():
        concact_list[1] = 0
        concact_list[3] = 0
    if (data2[1:-1]<1).all():
        concact_list[4] = 0
        concact_list[6] = 0
    if (data3[1:-1]<1).all():
        concact_list[5] = 0
        concact_list[7] = 0
    return concact_list


#quantity of interest    
def qoi(path,index_arr ,threshold=float('inf')):
    def list_to_array(my_list):
        my_array = np.array(my_list)
        shape = my_array.shape
        if len(shape) == 1:
            my_array = my_array[np.newaxis, :]
        return my_array
        
    ligs,circles,points,disp = [],[],[],[]
    index,linear,stiff_all,contact_all= [],[],[],[]

    for i in index_arr:
        
        try:
            ligs.append(np.loadtxt(os.path.join(path, '%s/a.dat'%i)))
            circles.append(np.loadtxt(os.path.join(path, '%s/circle.dat'%i)))    
            points.append(np.loadtxt(os.path.join(path, '%s/point.dat'%i)))

            stiffness = getStiff(path,i,threshold = threshold)
            contact = getContact(path,i)
            index.append(i)
               
        except Exception as e:
            #print('i=%d, warning:%s'%(i,e))
            continue
        if any(stiffness!=0):
            linear.append(1) 
            stiff_all.append(stiffness)   
            contact_all.append(contact)   
        else:
            linear.append(0)
        

    return {"index": list_to_array(index).reshape(-1,1),
        "ligs": list_to_array(ligs),
        "points": list_to_array(points),
        "circles":list_to_array(circles),
        "linear": list_to_array(linear).reshape(-1,1),
        "stiffness": list_to_array(stiff_all),
        "contact": list_to_array(contact_all)}

def add_qoi(qoi_list):
    return {
    "index": np.row_stack([q['index'] for q in qoi_list if q['index'].size!=0]),
    "ligs": np.row_stack([q['ligs'] for q in qoi_list if q['ligs'].size!= 0]),
    "points": np.row_stack([q['points'] for q in qoi_list if q['points'].size!= 0]),
    "circles":np.row_stack([q['circles'] for q in qoi_list if q['circles'].size!= 0]),
    "linear": np.row_stack([q['linear'] for q in qoi_list if q['linear'].size!= 0]),
    "stiffness": np.row_stack([q['stiffness'] for q in qoi_list if q['stiffness'].size!= 0])
    }

def get_features_arm0(config,circles,ligs):
    arm_config = config.arm_configs['arm0']
    ds = mix_space(arm_config.ds)
    assert len(ligs) == len(circles)
    features = []
    for i in range(len(ligs)):
        a = ligs[i]
        R,L,theta0,theta1,theta00,theta11 = circles[i]
        theta0 = np.radians(theta0)
        theta1 = np.radians(theta1)
        theta00 = np.radians(theta00)
        theta11 = np.radians(theta11)

        x0 = -L/2+R*math.cos(theta0)
        y0 = R*math.sin(theta0)
        x1 = -L/2+R*math.cos(theta1+theta0)
        y1 = R*math.sin(theta1+theta0)

        x00 = L/2+R*math.cos(theta00)
        y00 = R*math.sin(theta00)
        x11 = L/2+R*math.cos(theta11+theta00)
        y11 = R*math.sin(theta11+theta00)
        x = list(np.linspace(x0,x00,100)) 
        y = [sumf(c=a,f = ds.f, x=i) for i in x]
        features.append(x + y + [x1,y1,x0,y0,x00,y00,x11,y11])
    features = np.array(features)
    #features = (features-arm_config.feature_mean[0])/arm_config.feature_std[0]
    return features

def get_features_arm1(config,circles,ligs):
    arm_config = config.arm_configs['arm1']
    ds = mix_space(arm_config.ds)
    assert len(ligs) == len(circles)
    features = []
    for i in range(len(ligs)):
        a = ligs[i]
        R,L,theta0,theta1,theta00,theta11,r_s = circles[i]
        theta0 = np.radians(theta0)
        theta1 = np.radians(theta1)
        theta00 = np.radians(theta00)
        theta11 = np.radians(theta11)

        x0 = -L/2+R*math.cos(theta0)
        y0 = R*math.sin(theta0) 
        x1 = -L/2+R*math.cos(theta1+theta0)
        y1 = R*math.sin(theta1+theta0) 

        x00 = L/2+R*math.cos(theta00)
        y00 = R*math.sin(theta00)
        x11 = L/2+R*math.cos(theta11+theta00)
        y11 = R*math.sin(theta11+theta00)

        xs = -L/2+ (R+2*r_s)*np.cos(theta0)
        ys = (R+2*r_s)*np.sin(theta0)
        xss = L/2 + (R+2*r_s)*np.cos(theta00)
        yss = (R+2*r_s)*np.sin(theta00)

        x = list(np.linspace(xs,xss,100)) 
        y = [sumf(c=a,f = ds.f, x=i) for i in x]
        features.append(x + y + [x1,y1,x0,y0,x00,y00,x11,y11])
    features = np.array(features)
    #features = (features-arm_config.feature_mean[0])/arm_config.feature_std[0]
    return features

def get_features_arm2(config,circles,ligs):
    arm_config = config.arm_configs['arm2']
    ds = mix_space(arm_config.ds)
    assert len(ligs) == len(circles)
    features = []
    for i in range(len(ligs)):
        a = ligs[i]
        R,L,theta0,theta1,theta00,theta11 = circles[i]
        theta0 = np.radians(theta0)
        theta1 = np.radians(theta1)
        theta00 = np.radians(theta00)
        theta11 = np.radians(theta11)

        x0 = -L/2+R*math.cos(theta0)
        y0 = R*math.sin(theta0)
        x1 = -L/2+R*math.cos(theta1+theta0)
        y1 = R*math.sin(theta1+theta0)

        x00 = L/2+R*math.cos(theta00)
        y00 = R*math.sin(theta00)
        x11 = L/2+R*math.cos(theta11+theta00)
        y11 = R*math.sin(theta11+theta00)

        x = list(np.linspace(x0,x00,100)) 
        y = [sumf(c=a,f = ds.f, x=i) for i in x]
        features.append(x + y + [x1,y1,x0,y0,x00,y00,x11,y11])
    features = np.array(features)
    #features = (features-arm_config.feature_mean[0])/arm_config.feature_std[0]
    return features

def get_features_arm3(config,circles,ligs):
    arm_config = config.arm_configs['arm3']
    ds = mix_space(arm_config.ds)
    assert len(ligs) == len(circles)
    features = []
    for i in range(len(ligs)):
        a = ligs[i]
        R,L,theta0,theta1,theta00,theta11,r_s = circles[i]
        theta0 = np.radians(theta0)
        theta1 = np.radians(theta1)
        theta00 = np.radians(theta00)
        theta11 = np.radians(theta11)

        x0 = -L/2+R*math.cos(theta0)
        y0 = R*math.sin(theta0)
        x1 = -L/2+R*math.cos(theta1+theta0)
        y1 = R*math.sin(theta1+theta0)

        x00 = L/2+R*math.cos(theta00)
        y00 = R*math.sin(theta00)
        x11 = L/2+R*math.cos(theta11+theta00)
        y11 = R*math.sin(theta11+theta00)

        xs = -L/2+ (R+2*r_s)*np.cos(theta0)
        ys = (R+2*r_s)*np.sin(theta0)

        x = list(np.linspace(xs,x00,100)) 
        y = [sumf(c=a,f = ds.f, x=i) for i in x]
        features.append(x + y + [x1,y1,x0,y0,x00,y00,x11,y11])
    features = np.array(features)
    #features = (features-arm_config.feature_mean[0])/arm_config.feature_std[0]
    return features

#this function get the coefficients of the subfunctions of the ligaments shape of each design spaces separatly
def get_ligs_arm0(config,circles,points):
    assert len(points) == len(circles)
    arm_config = config.arm_configs['arm0']
    
    ligs = []
    for i in range(len(points)):
        chiral = arm0_one_ligament_asym(R=circles[i][0],
                                   L=circles[i][1],
                                   theta0 = circles[i][2], 
                                   theta1=circles[i][3], 
                                   theta00 = circles[i][4], 
                                   theta11=circles[i][5], 
                                   n = arm_config.lig_degree,
                                   ds = mix_space(arm_config.ds))
        a,_ = chiral.design_lig_points(points = points[i].reshape(-1,2))
        ligs.append(a)
    return ligs

def get_ligs_arm1(config,circles,points):
    assert len(points) == len(circles)
    arm_config = config.arm_configs['arm1']
    ligs = []
    for i in range(len(points)):
        chiral = arm1_one_ligament_asym(R=circles[i][0],
                                   L=circles[i][1],
                                   theta0 = circles[i][2], 
                                   theta1=circles[i][3], 
                                   theta00 = circles[i][4], 
                                   theta11=circles[i][5], 
                                   r_s=circles[i][6],
                                   n = arm_config.lig_degree,
                                   ds = mix_space(arm_config.ds))
        a,_ = chiral.design_lig_points(points = points[i].reshape(-1,2))
        ligs.append(a)
    return ligs

def get_ligs_arm2(config,circles,points):
    assert len(points) == len(circles)
    arm_config = config.arm_configs['arm2']
    
    ligs = []
    for i in range(len(points)):
        chiral = arm0_one_ligament_asym(R=circles[i][0],
                                   L=circles[i][1],
                                   theta0 = circles[i][2], 
                                   theta1=circles[i][3], 
                                   theta00 = circles[i][4], 
                                   theta11=circles[i][5], 
                                   n = arm_config.lig_degree,
                                   ds = mix_space(arm_config.ds))
        a,_ = chiral.design_lig_points(points = points[i].reshape(-1,2))
        ligs.append(a)
    return ligs

def get_ligs_arm3(config,circles,points):
    assert len(points) == len(circles)
    arm_config = config.arm_configs['arm3']
    ligs = []
    for i in range(len(points)):
        chiral = arm3_one_ligament_asym(R=circles[i][0],
                                   L=circles[i][1],
                                   theta0 = circles[i][2], 
                                   theta1=circles[i][3], 
                                   theta00 = circles[i][4], 
                                   theta11=circles[i][5], 
                                   r_s=circles[i][6],
                                   n = arm_config.lig_degree,
                                   ds = mix_space(arm_config.ds))
        a,_ = chiral.design_lig_points(points = points[i].reshape(-1,2))
        ligs.append(a)
    return ligs

#pipelines
def get_train_data(objective, config, it):
    path_obejective = os.path.join(config.workdir, objective)
    file_to_read = open(os.path.join(path_obejective, config.path_data, 'counts.pkl'), 'rb')
    counts = pickle.load(file_to_read)
    savepath = os.path.join(path_obejective,config.path_data)
    features_all = []
    for arm_name in config.arm_names:
        get_features_high = globals()[f'get_features_{arm_name}']
        #iter = 0 
        data_list = []
        for it_curr in range(it+1):
            if it_curr == 0: all_data = qoi(path = os.path.join(config.initial_data_path, '%s_samples'%arm_name), index_arr = range(config.n_initial), threshold = config.threshold)
            #else: all_data = qoi(path=os.path.join(path_obejective,arm_name),index_arr = range(config.n_initial, counts[it][arm_name]), threshold = config.threshold)
            else: all_data = qoi(path=os.path.join(path_obejective,arm_name),index_arr = range(config.n_initial, config.n_initial + config.n_next*it_curr), threshold = config.threshold)
            data_list.append(all_data)
        all_data = add_qoi(data_list)

        ligs = all_data['ligs'][all_data['index']][all_data['linear']==1]
        points = all_data['points'][all_data['index']][all_data['linear']==1]
        circles = all_data['circles'][all_data['index']][all_data['linear']==1]
        Xtrain = get_features_high(config,circles,ligs)
        Ytrain = all_data['stiffness']

        ##data augmentation
        if arm_name in ['arm0','arm2']:
            circles_aug = np.concatenate([circles[:,0:2],circles[:, 4:5]-180, circles[:, 5:6],circles[:, 2:3]+180,circles[:, 3:4]],axis =1)
        else:
            circles_aug = np.concatenate([circles[:,0:2],circles[:, 4:5]-180, circles[:, 5:6],circles[:, 2:3]+180,circles[:, 3:4],circles[:, 6:7]],axis =1)
        
        if arm_name == 'arm3':
            ligs_aug = get_ligs_arm3aug(config, circles_aug,-points)
            Xtrain_aug_1 = get_features_arm3aug(config,circles_aug,ligs_aug)
            Xtrain_aug = np.row_stack([Xtrain,Xtrain_aug_1])
            Ytrain_aug = np.row_stack([Ytrain,Ytrain])
        else:
            get_ligs = globals()[f'get_ligs_{arm_name}']
            ligs_aug = get_ligs(config, circles_aug,-points)
            Xtrain_aug_1 = get_features_high(config,circles_aug,ligs_aug)


        Xtrain_aug = np.row_stack([Xtrain,Xtrain_aug_1])
        Ytrain_aug = np.row_stack([Ytrain,Ytrain])
        assert Xtrain_aug.shape[0] == Ytrain_aug.shape[0]
        features_all.append(np.column_stack([Xtrain_aug, Ytrain_aug]))   
    features_all =  np.row_stack(features_all)
    np.savetxt(os.path.join(savepath,"stiffness_iter%s.dat"%it),features_all)
    ##save objective values
    
    x_features = features_all[:,:-8]
    stiffness = features_all[:,-8:]
    for obj_i in range(len(objective)//6):
        objective_curr = objective[obj_i*6:(obj_i+1)*6]
        r = [config.Sindex[objective_curr [:3]], config.Sindex[objective_curr [3:]]]
        label = np.log(np.abs(stiffness[:,r[0]]/stiffness[:,r[1]]))
        x_features = np.column_stack([x_features,label])
    np.savetxt(os.path.join(savepath,"objective_iter%s.dat"%it), x_features)
    #save all non-reciprocal and asymmetric properties

def write_data(ligs,circles,points,path,start=0):
    for i in range(len(ligs)):
        n = i+start
        np.savetxt(os.path.join(path, 'a%s.dat'%n),ligs[i])
        np.savetxt(os.path.join(path, 'circle%s.dat'%n),circles[i])
        np.savetxt(os.path.join(path, 'point%s.dat'%n),points[i])
        for di,disp in enumerate([[-0.08,0.0], [0.0,-0.08], [0.08,0.0], [0.0,0.08]]):
            np.savetxt(os.path.join(path, 'disp%s_test%s.dat'%(n,di)),disp)


def sampling(config):
    for arm_name in config.arm_names:
        arm_config = config.arm_configs[arm_name]
        asym_poly_designs = globals()[f'{arm_name}_asym_poly_designs']
        ligs, circles, points = asym_poly_designs(n_designs=arm_config.n_nextpool, n=arm_config.lig_degree, ds_name=arm_config.ds)
        arm = {'ligs':ligs, 'circles':circles,'points':points}
        if not os.path.exists(config.path_population): os.makedirs(config.path_population)
        f = open(os.path.join(config.path_population,arm_name+'.pkl'),'wb')
        pickle.dump(arm,f)
        f.close()

def test_features(config):
    for arm_name in config.arm_names:
        f = open(os.path.join(config.path_population,arm_name+'.pkl'),'rb')
        arm = pickle.load(f)
        get_features_high = globals()[f'get_features_{arm_name}']
        features = get_features_high(config, circles = arm["circles"], ligs = arm['ligs'])
        np.savetxt(os.path.join(config.path_population,"%s.dat"%arm_name), features)

def initial_prepare(objective, config):
    #iteration=0
    counts = {0:{}}
    if not os.path.exists(config.path_population): os.mkdir(config.path_population)
    for arm_name in config.arm_names:
        arm_config = config.arm_configs[arm_name]
        counts[0][arm_name] = config.n_initial
        simulation_path = os.path.join(config.workdir,objective,arm_name)
        Path(simulation_path).mkdir(parents=True, exist_ok=True)
        #asym_poly_designs = globals()[f'{arm_name}_asym_poly_designs']
        #ligs, circles, points = asym_poly_designs(n_designs=config.n_initial, n = arm_config.lig_degree, ds_name=arm_config.ds)
        #write_data(ligs=ligs, circles=circles, points=points, path=simulation_path, start=0)
    f = open(os.path.join(config.workdir, objective, config.path_data,'counts.pkl'),'wb')
    pickle.dump(counts,f)
    f.close

def next_simulation(objective, config,it):
    assert it > 0
    path_log  = os.path.join(config.workdir,objective, "./logs/")
    logger = get_logger(os.path.join(path_log,'optimization_history.log'),display=False, save = True)
    logger.info('iteration: {}'.format(it))
    logger.info('preparing for the input files for simulation')
    
    file_to_read = open(os.path.join(config.workdir,objective,config.path_data, 'index_labeled.pkl'),'rb')
    arm_select = pickle.load(file_to_read)[it]

    file_to_read = open(os.path.join(config.workdir, objective, config.path_data,'counts.pkl'),'rb')
    counts = pickle.load(file_to_read)
    counts[it] = {}

    for arm_name in config.arm_names:
        simulation_path = os.path.join(config.workdir,objective,arm_name)
        file_to_read = open(os.path.join(config.path_population,arm_name+'.pkl'),'rb')
        arm_designs = pickle.load(file_to_read)
        select_index = arm_select[arm_name]
        ligs = [arm_designs['ligs'][k] for k in select_index]
        circles = [arm_designs['circles'][k] for k in select_index]
        points = [arm_designs['points'][k] for k in select_index]
        #ligs, circles, points = arm_designs['ligs'][select_index], arm_designs['circles'][select_index], arm_designs['points'][select_index]        
        write_data(ligs=ligs, circles=circles, points=points, path=simulation_path, start=counts[it-1][arm_name])
        counts[it][arm_name] = counts[it-1][arm_name] + select_index.shape[0]
    f = open(os.path.join(config.workdir, objective, config.path_data,'counts.pkl'),'wb')
    pickle.dump(counts,f)
    f.close
    logger.info('counts: {}'.format(counts))



def show_path(path, index, sym=True):
    all_data =qoi(path,[index])
    stiff = all_data['stiff'][0]
    for i in range(4):
        try:
            data = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test%s_output.dat'%i))
            cpress = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test%s_outputcp.dat'%i))
            fs,us = collect_data(data)    
            gt_plot(data,cpress,mc = stiff[:4].reshape(2,2), mn = stiff[4:].reshape(2,2))
        except:
            continue
            
def show_info(arm_name,path,index,sf_list = ['kxx-','kxy-','kyx-','kyy-','kxx+','kxy+','kyx+','kyy+']):
    qoi_curr= qoi(path,range(index,index+1))
    print('ligs:',qoi_curr['ligs'])
    print('circles:',qoi_curr['circles'])
    print('stiffness:','\n',qoi_curr['stiffness'][:,:4],'\n',qoi_curr['stiffness'][:,4:])
    print('contact:',qoi_curr['contact'])

    data = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test0_output.dat'))
    fs,us = get_fs_us(data)
    fs,us = np.abs(fs), np.abs(us)
    if 'kxx-' in sf_list: plt.scatter(us[:,0],fs[:,0],marker = 'o',c='r',s=30,label = 'kxx-')
    if 'kyx-' in sf_list: plt.scatter(us[:,0],fs[:,1],marker = 's',c = 'r',s=30, label = 'kyx-')
    
    data = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test1_output.dat'))
    fs,us = get_fs_us(data)
    fs,us = np.abs(fs), np.abs(us)
    if 'kxy-' in sf_list: plt.scatter(us[:,1],fs[:,0],marker='P', c= 'r', s=30, label = 'kxy-')
    if 'kyy-' in sf_list: plt.scatter(us[:,1],fs[:,1],marker = '^',c ='r', s=30, label = 'kyy-')
    

    data = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test2_output.dat'))
    fs,us = get_fs_us(data)
    fs,us = np.abs(fs), np.abs(us)
    if 'kxx+' in sf_list: plt.scatter(us[:,0],fs[:,0],marker = 'o', s= 30, facecolors='none', edgecolors='k', label = 'kxx+')
    if 'kyx+' in sf_list: plt.scatter(us[:,0],fs[:,1],marker = 's', s = 30, facecolors='none', edgecolors='k',label = 'kyx+')
    
    data = np.loadtxt(os.path.join(path,str(index), 'one_ligament_test3_output.dat'))
    fs,us = get_fs_us(data)
    fs,us = np.abs(fs), np.abs(us)
    if 'kxy+' in sf_list: plt.scatter(us[:,1],fs[:,0],marker='P',s= 30, facecolors='none', edgecolors='k', label = 'kxy+')
    if 'kyy+' in sf_list: plt.scatter(us[:,1],fs[:,1],marker = '^',s= 30, facecolors='none', edgecolors='k', label = 'kyy+')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index(i) for i in sf_list]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = (1.01,0))
    
    show_geo = globals()[f'{arm_name}_show_geo_abaqus']
    show_geo(path, index)
        
    plt.show()