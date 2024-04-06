import numpy as np
import pandas as pd
import logging
from numpy.linalg import lstsq 
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os 


def get_logger(logpath, display=True, save = True, name = None):
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    if save:
        file_handler = logging.FileHandler(logpath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if display:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def collect_data(data):
    RFs_test1 = []
    Us_test1 = []
    for i in range(3):

        f = {'step0':np.zeros(data[:,3:6][:,i].shape[0]),
            'step1':data[:,3:6][:,i],
             'step2':data[:,9:12][:,i],
             'step3':data[:,15:18][:,i],
             'step4':data[:,21:24][:,i]    
        }

        u = {'step0':np.zeros(data[:,3:6][:,i].shape[0]),
            'step1':data[:,:3][:,i],
             'step2':data[:,6:9][:,i],
             'step3':data[:,12:15][:,i],
             'step4':data[:,18:21][:,i]    
        }

        RFs_test1.append(pd.DataFrame(f))
        Us_test1.append(pd.DataFrame(u))

    f1 = np.concatenate([RFs_test1[0]['step1'] ,RFs_test1[0]['step2'],RFs_test1[0]['step3'],RFs_test1[0]['step4']])
    f2 = np.concatenate([RFs_test1[1]['step1'] ,RFs_test1[1]['step2'],RFs_test1[1]['step3'],RFs_test1[1]['step4']])
    #f3 = np.concatenate([RFs_test1[2]['step1'] ,RFs_test1[2]['step2'],RFs_test1[2]['step3'],RFs_test1[2]['step4']])

    u1 = np.concatenate([Us_test1[0]['step1'] ,Us_test1[0]['step2'],Us_test1[0]['step3'],Us_test1[0]['step4']])
    u2 = np.concatenate([Us_test1[1]['step1'] ,Us_test1[1]['step2'],Us_test1[1]['step3'],Us_test1[1]['step4']])
    #u3 = np.concatenate([Us_test1[2]['step1'] ,Us_test1[2]['step2'],Us_test1[2]['step3'],Us_test1[2]['step4']])

    #fs = 5/(4*20*30)*np.array([f1, f2, f3]).T
    #us = 1/(20)*np.array([u1, u2, u3]).T

    fs = np.array([f1, f2]).T
    us = np.array([u1, u2]).T
    
    return fs,us

def plot_path(data, data_density = 41):
    fs,us = collect_data(data)
    rf1=fs[:,0]
    rf2=fs[:,1]
    u1=us[:,0]
    u2=us[:,1]
    
    fig, axs = plt.subplots(1,5, figsize = (20,4))
    axs[0].scatter(u1,rf1,s=10)
    axs[1].scatter(u1,rf2,s=10)
    for i in range(2):
        axs[i].set_xlabel('u1(mm)')

    axs[2].scatter(u2,rf1,s=10)
    axs[3].scatter(u2,rf2,s=10)
    for i in range(2,4):
        axs[i].set_xlabel('u2(mm)')
    
    anotate_path(axs,fs,us,data_density=data_density)    

    for i in [0,2]:
        axs[i].set_ylabel('RF1(N)')

    for i in [1,3]:
        axs[i].set_ylabel('RF2(N)')
    
    #strain energy
    axs[4].scatter(np.array(range(data_density*4))/(data_density*4)*4, 
            np.concatenate([data[:,-4:-3],data[:,-3:-2],data[:,-2:-1],data[:,-1:]],axis =0),s=10)
    axs[4].set_xlabel('time')
    axs[4].set_ylabel('strain energy')
    axs[4].set_xticks((1,2,3,4))

    #plt.subplots_adjust(top=0.7)
    #plt.subplots_adjust(left=-0.3)
    fig.tight_layout()
    plt.show()

    
def plot_path_animation(data,name,data_density = 21):
    fs,us = collect_data(data)
    
    rf1=fs[:,0]
    rf2=fs[:,1]
    u1=us[:,0]
    u2=us[:,1]
    
    fig, axs = plt.subplots(1,4, figsize = (15,4))
    def animation_func(i):
        #plt.scatter(u1[i],rf1[i],c = 'black')
        #plt.xlim(u1.min()-0.01,u1.max()+0.01)
        #plt.ylim(rf1.min()-500,rf1.max()+500)
    
        axs[0].scatter(u1[i],rf1[i],c = 'black',s=20)
        axs[0].set_xlim(u1.min()-0.01,u1.max()+0.01)
        axs[0].set_ylim(rf1.min()-500,rf1.max()+500)
        
        axs[1].scatter(u1[i],rf2[i],c = 'black',s=20)
        axs[1].set_xlim(u1.min()-0.01,u1.max()+0.01)
        axs[1].set_ylim(rf2.min()-500,rf2.max()+500)
        
        axs[2].scatter(u2[i],rf1[i],c = 'black',s=20)
        axs[2].set_xlim(u2.min()-0.01,u2.max()+0.01)
        axs[2].set_ylim(rf1.min()-500,rf1.max()+500)
        
        axs[3].scatter(u2[i],rf2[i],c = 'black',s=20)
        axs[3].set_xlim(u2.min()-0.01,u2.max()+0.01)
        axs[3].set_ylim(rf2.min()-500,rf2.max()+500)
    
    for i in range(2):
        axs[i].set_xlabel('U1(mm)')
        
    for i in range(2,4):
        axs[i].set_xlabel('U2(mm)')
    
    anotate_path(axs,fs,us,data_density=data_density)    
    
    for i in [0,2]:
        axs[i].set_ylabel('RF1(N)')

    for i in [1,3]:
        axs[i].set_ylabel('RF2(N)')

    #plt.subplots_adjust(top=0.7)
    #plt.subplots_adjust(left=-0.3)
    fig.tight_layout()
    animation = FuncAnimation(fig, animation_func, interval = 100, frames = np.arange(0,len(u1), 1))
    animation.save(name + '.gif')

def anotate_path(axs,fs,us,data_density=41):
    rf1=fs[:,0]
    rf2=fs[:,1]
    u1=us[:,0]
    u2=us[:,1]
    
    for i in range(4):
        axs[0].annotate(i, (u1[0+data_density*i],rf1[0+data_density*i]), fontsize = 20,alpha = 0.6)
        axs[1].annotate(i, (u1[0+data_density*i],rf2[0+data_density*i]), fontsize = 20,alpha = 0.6)
        axs[2].annotate(i, (u2[0+data_density*i],rf1[0+data_density*i]), fontsize = 20,alpha = 0.6)
        axs[3].annotate(i, (u2[0+data_density*i],rf2[0+data_density*i]), fontsize = 20,alpha = 0.6)
        
        
mc = np.array([[86182, 48581],[48408, 28514]])
mn = np.array([[41187.56, 20248.52],[20248.52, 10377.73]])
def gt_plot(data0,cpress0,mc = mc, mn = mn):
    fs, us = collect_data(data0)
    us_temp,fs_temp  = us[1:-1], fs[1:-1]
    cindex = contact_point(data0,cpress0)
    
    fig,axs = plt.subplots(1,4, figsize = (15,3.5))

    k = 0
    for i, j in (0,0),(0,1),(1,0),(1,1):
        #axs[k].scatter(us_temp[:,i], us_temp@mc[:,j],c = 'r',s = 10) 
        axs[k].plot(us[:,i], us@mc[j,:].T,c = 'r') 
        #axs[k].scatter(us_temp[:,i], us_temp@mn[:,j],marker = '*',c = 'k', s = 20) 
        axs[k].plot(us[:,i], us@mn[j,:].T,c = 'k') 
        axs[k].scatter(us[:,i], fs[:,j],s = 10, c = 'b')
        axs[k].scatter(us[cindex,i],fs[cindex,j],color = 'red')
        axs[k].set_ylabel(['F1','F2'][j])
        axs[k].set_xlabel(['U1','U2'][i])
        k+=1
    
    anotate_path(axs,fs,us)

    fig.tight_layout()
    plt.show()
    
    return fig,axs

def contact_point(data,cpress):
    fs, us = collect_data(data)
    us_temp = us[1:-1]
    fs_temp = fs[1:-1]
    if(np.argmax(cpress[1:-1]<1) == 0):   
        cpoint=np.argmax(cpress[1:-1]>1)+1
    else:
        cpoint=np.argmax(cpress[1:-1]<1)+1
    #u1c_temp = us[cpoint,0]
    #u2c_temp = us[cpoint,1]
    return cpoint

