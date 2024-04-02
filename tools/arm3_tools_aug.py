import numpy as np
from numpy.linalg import inv
import random
from matplotlib import pyplot as plt
##chiral Data
import os,sys
sys.path.append("../tools")
sys.path.append("..")
from lig_space import *

import ml_collections
from configs.configs_default import get_default_config
configAll = get_default_config()
config = configAll.arm_configs['arm3']

class arm3aug_one_ligament_asym():
    """
    A class for designing an asymmetric ligament.

    Attributes:
    - R: float, the radius of the circle
    - L: float, the distance betwee two circles
    - theta0: float, the angle of the tangent at the start point(ligament on the left circle)
    - theta1: float, the angle of the tangent at the end point(ligament on the left circle)
    - theta00: float, the angle of the tangent at the start point(ligament on the right circle)
    - theta11: float, the angle of the tangent at the end point(ligament on the right circle)
    - n: int, the number of coefficients for the ligament
    - ds: list, the mix space of design functions for the ligament, the element of the list can be 'poly' or 'sin'. 
    """
    def __init__(self, R=5, L=20,theta0 = -60, theta1=-60, theta00 = 130, theta11=-70, r_s = 0.5, n = 10,ds = mix_space(['poly']*10)):
        self.L = L
        self.R = R
        self.theta0 = np.radians(theta0)
        self.theta1 = np.radians(theta1)
        self.theta00 = np.radians(theta00)
        self.theta11 = np.radians(theta11)
        self.x0 = -L/2+ R*np.cos(self.theta0)
        self.y0 = R*np.sin(self.theta0)
        self.y0d =  -1/np.tan(self.theta0)
        self.y0dd = -R**2/self.y0**3
        self.x00 = L/2 + R*np.cos(self.theta00)
        self.y00 = R*np.sin(self.theta00)
        self.y00d =  -1/np.tan(self.theta00)
        self.y00dd = -R**2/self.y00**3
        self.n = n
        self.ds = ds

        self.r_s = r_s 
        self.x1 = -L/2+ R*np.cos(self.theta0)
        self.y1 = R*np.sin(self.theta0)
        self.x11 = L/2 + (R+2*r_s)*np.cos(self.theta00)
        self.y11 = (R+2*r_s)*np.sin(self.theta00)


    def design_lig(self,candidates = range(10),seed = None):
        """
        Design an asymmetric ligament by random polynomial function.

        Args:
        - candidates: range or list of int, the indices of coefficients to be randomized
        - seed: int, the random seed

        Returns:
        - list of float, the coefficients for the ligament
        """

        candidates = list(candidates)
        x0,x00,R,L,n = self.x1, self.x11, self.R, self.L,self.n
        ds = self.ds
        
        if n < 3: return
        if seed: random.seed(seed)
            
        
        #random generate coeff for items in combination
        a = [0]*n
        # May need to reconsider the constrain on a
        for i in candidates:
            #a[i] = random.uniform(0,L**(-2*(i-1)))*random.choice([1,-1])
            #a[i] = random.uniform(0,R*2**(2*i+15)/L**(2*i+1))*random.choice([1,-1])
            if ds.name[i] == 'poly':
                a[i] = random.uniform(0,1/2**i)*random.choice([1,-1])
            elif ds.name[i] == 'sin':
                a[i] = random.uniform(0,1)*random.choice([1,-1])
        
        # select three coeff to be solved based on boundary condition, leave other coeff as random
        i,j,k,ii,jj,kk = random.sample(candidates,6)  
        #obtain the coeff for i,j,k based on boundary condition(continuity in 0,1,2 order derivative)
        
        a11,a12,a13,a14,a15,a16 = ds.f(i,x0), ds.f(j,x0), ds.f(k,x0), ds.f(ii,x0), ds.f(jj,x0), ds.f(kk,x0)
        a21,a22,a23,a24,a25,a26 = ds.f_diff(i,x0), ds.f_diff(j,x0), ds.f_diff(k,x0),ds.f_diff(ii,x0), ds.f_diff(jj,x0), ds.f_diff(kk,x0)
        a31,a32,a33,a34,a35,a36 = ds.f_diff2(i,x0), ds.f_diff2(j,x0), ds.f_diff2(k,x0),ds.f_diff2(ii,x0), ds.f_diff2(jj,x0), ds.f_diff2(kk,x0)
        
        a41,a42,a43,a44,a45,a46 = ds.f(i,x00), ds.f(j,x00), ds.f(k,x00), ds.f(ii,x00), ds.f(jj,x00), ds.f(kk,x00)
        a51,a52,a53,a54,a55,a56 = ds.f_diff(i,x00), ds.f_diff(j,x00), ds.f_diff(k,x00),ds.f_diff(ii,x00), ds.f_diff(jj,x00), ds.f_diff(kk,x00)
        a61,a62,a63,a64,a65,a66 = ds.f_diff2(i,x00), ds.f_diff2(j,x00), ds.f_diff2(k,x00),ds.f_diff2(ii,x00), ds.f_diff2(jj,x00), ds.f_diff2(kk,x00)
        
        g0 = sumf(c=a,f=ds.f,x = x0) - a11*a[i] - a12*a[j] - a13*a[k] - a14*a[ii] - a15*a[jj] - a16*a[kk]
        g0d = sumf(c=a,f=ds.f_diff,x = x0) - a21*a[i] - a22*a[j] - a23*a[k] - a24*a[ii] - a25*a[jj] - a26*a[kk]
        g0dd = sumf(c=a,f=ds.f_diff2,x = x0) - a31*a[i] - a32*a[j]- a33*a[k] - a34*a[ii] - a35*a[jj] - a36*a[kk]
        
        g1 = sumf(c=a,f=ds.f,x = x00) - a41*a[i] - a42*a[j] - a43*a[k]- a44*a[ii] - a45*a[jj] - a46*a[kk]
        g1d = sumf(c=a,f=ds.f_diff,x = x00) - a51*a[i] - a52*a[j]- a53*a[k] - a54*a[ii] - a55*a[jj] - a56*a[kk]
        g1dd = sumf(c=a,f=ds.f_diff2,x = x00) - a61*a[i] - a62*a[j]- a63*a[k]- a64*a[ii] - a65*a[jj] - a66*a[kk]
        
        A=[[a11,a12,a13,a14,a15,a16],
           [a21,a22,a23,a24,a25,a26],
           [a31,a32,a33,a34,a35,a36],
           [a41,a42,a43,a44,a45,a46],
           [a51,a52,a53,a54,a55,a56],
           [a61,a62,a63,a64,a65,a66],]
        b = [self.y0-g0, self.y0d-g0d,self.y0dd-g0dd, self.y00-g1, self.y00d-g1d,self.y00dd-g1dd]
    
        inv_A = np.linalg.inv(A)
        a[i], a[j], a[k], a[ii],a[jj],a[kk] = inv_A@b
        return a

    def pick(self,seed = None):
        x0,x00,R,L,n =  self.x1, self.x11, self.R, self.L,self.n
        if n < 6: return None
        if seed: random.seed(seed)
        n_points = n-6
        #pick random n_points:
        def low(x):
            if x >= L/2-R:
                return np.sqrt(R**2-(x-L/2)**2)
            return -R
        def high(x):
            if x <= -L/2+R:
                return -np.sqrt(R**2-(x+L/2)**2)
            return R

        x_picked = [random.uniform(x0, x00) for i in range(n_points)]
        y_picked = [random.uniform(low(i),high(i)) for i in x_picked]
        while x0 in x_picked:
            x_picked = [random.uniform(x0, x00) for i in range(n_points)]
            y_picked = [random.uniform(low(i),high(i)) for i in x_picked]
        return x_picked, y_picked

    def design_lig_points(self,candidates = None, points = None):
        candidates = list(range(self.n)) if candidates is None else list(candidates)
        x0,x00,n =  self.x1, self.x11,self.n
        ds = self.ds
        if points is None:
            x_picked, y_picked = self.pick()
            points = np.column_stack([x_picked,y_picked])
        else:
            x_picked,y_picked = list(points[:,0]), list(points[:,1])
        x_picked = np.append(x_picked, [x0,x00])
        y_picked = np.append(y_picked,[self.y1,self.y11])

        B = np.ones((n,n))
        for i in range(x_picked.shape[0]):
            B[i] = [ds.f(j, x_picked[i]) for j in candidates]
        B[-4] = [ds.f_diff(j, x0) for j in candidates]
        B[-3] = [ds.f_diff(j, x00) for j in candidates]
        B[-2] = [ds.f_diff2(j, x0) for j in candidates]
        B[-1] = [ds.f_diff2(j, x00) for j in candidates]

        y_picked = np.append(y_picked, [self.y0d,self.y00d, self.y0dd, self.y00dd])
        inv_B = np.linalg.inv(B)
        a = np.zeros(max(candidates)+1)
        a[candidates] = inv_B@y_picked
        #return inv_B@y_picked, points
        return a, points

    def plot_geo(self,a, points = None):
        color_lig = '#265894'
        theta0,theta1,x0 = self.theta0, self.theta1, self.x1
        theta00,theta11,x00 = self.theta00,self.theta11,self.x11
        R,L,n = self.R,self.L,self.n
        ds = self.ds
        
        plt.figure().set_figwidth(9)
        #circle
        theta = np.linspace(np.radians(0),np.radians(360),50)
        x_circle = -L/2+R*np.cos(theta)
        y_circle = R*np.sin(theta)
        plt.plot(x_circle, y_circle, c='k')
        plt.plot(-x_circle, -y_circle, c='k')
            
        #ligament on circle
        theta_left = np.linspace(theta0,theta0+theta1,50)
        x_ligleft = -L/2+R*np.cos(theta_left)
        y_ligleft = R*np.sin(theta_left)

        theta_right = np.linspace(theta00,theta00+theta11,50)
        x_ligright = L/2+R*np.cos(theta_right)
        y_ligright = R*np.sin(theta_right)

        #plt.scatter(x_ligleft, y_ligleft, c = '#265894',s=8)
        #plt.scatter(x_ligright, y_ligright, c = '#265894',s=8)
        x_lig = [np.flip(x_ligleft),x_ligright]
        y_lig = [np.flip(y_ligleft),y_ligright]
        
        #ligament on the small circle

        theta = np.linspace(theta00,theta00 - np.radians(180),25)
        x_central = L/2+ (R+self.r_s)*np.cos(self.theta00)
        y_central = (R+self.r_s)*np.sin(self.theta00)
        x_circle1 = x_central + self.r_s*np.cos(theta)
        y_circle1 = y_central + self.r_s*np.sin(theta)
        #plt.scatter(x_circle1, y_circle1,c= '#265894',s=10)
        x_lig.insert(1,x_circle1)
        y_lig.insert(1,y_circle1)

        plt.plot([-10,x_ligleft[-1]], [0,y_ligleft[-1]],'--',c='k',linewidth=0.5,dashes=(6, 6))
        plt.plot([-10,x_ligleft[0]], [0,y_ligleft[0]],'--',c='k',linewidth=0.5,dashes=(6, 6))
        plt.plot([10,x_ligright[-1]], [0,y_ligright[-1]],'--',c='k',linewidth=0.5,dashes=(6, 6))
        plt.plot([10,x_ligright[0]], [0,y_ligright[0]],'--',c='k',linewidth=0.5,dashes=(6, 6))

        #ligament not on circle
        x = np.linspace(x0, x00, 200)
        y = np.array([sumf(c=a, f=ds.f, x=i) for i in x])
        x_lig.insert(1,x)
        y_lig.insert(1,y)
        plt.plot([i for item in x_lig for i in item], [i for item in y_lig for i in item], c = color_lig,linewidth=3,zorder=8)

        if points is not None:
            plt.scatter(points[:,0],points[:,1],s = 50,c='red',marker = "*",zorder=9)

        plt.scatter(x_ligleft[-1], y_ligleft[-1], c = '#265894',s=40,zorder=10)
        plt.scatter(x_ligright[-1], y_ligright[-1], c = '#265894',s=40,zorder=10)

        plt.axis('scaled')
        plt.show()


    def valid_lig(self,a):
        theta0, theta00= self.theta0, self.theta00
        x0, x00 = self.x1, self.x11
        R,L,n = self.R,self.L,self.n
        ds = self.ds
        
        x_all = list(np.linspace(x0,x00,100))
        y_all = np.abs([sumf(c = a,f = ds.f, x = i) for i in x_all])
        y_all_diff = np.abs([sumf(c = a,f = ds.f_diff, x = i) for i in x_all])
        
        theta_left = np.linspace(theta0,0, 200)[1:-1]
        x_left = -L / 2 + R * np.cos(theta_left)
        y_circle_left = R * np.sin(theta_left)
        y_ligament_left = np.array([sumf(c=a, f=ds.f, x=i) for i in x_left])

        theta_right = np.linspace(theta00,np.pi,200)[1:-1]
        x_right = L / 2 + R * np.cos(theta_right)
        y_circle_right = R * np.sin(theta_right)
        y_ligament_right = np.array([sumf(c=a, f=ds.f, x=i) for i in x_right])

        #print(y_circleleft-y_ligleft)
        if np.any(y_circle_left<=y_ligament_left) or np.any(y_circle_right>=y_ligament_right) or max(y_all)>=R or max(y_all_diff)>=50:
            return False
        return True

#generate asymmetric ligament
def arm3aug_asym_poly_designs(n_designs, n=8,ds_name=['poly']*10,points_fix=None):
    ligs = []
    circles = []
    points = []
    while len(ligs)<n_designs:
        theta0 = random.uniform(-90,0)
        theta1 = random.uniform(-90,-20)
        theta00 = random.uniform(90,180)
        theta11 = random.uniform(20,90)
        R = random.uniform(3,7)
        r_s = random.uniform(0.1,R/10)

        chiral = arm3aug_one_ligament_asym(R=R, L=20,theta0 = theta0, theta1=theta1, theta00 = theta00, theta11=theta11, r_s=r_s, n = n, ds = mix_space(ds_name))
        #a = chiral.design_lig(candidates = range(chiral.n))
        try:
            a, point = chiral.design_lig_points(points = points_fix)
        except:
            continue
        if chiral.valid_lig(a):
            #chiral.plot_geo(a,point)
            ligs.append(a)
            points.append(point)
            circles.append([chiral.R,chiral.L, math.degrees(chiral.theta0), math.degrees(chiral.theta1),
                            math.degrees(chiral.theta00), math.degrees(chiral.theta11),chiral.r_s])
    return ligs, circles, points

def arm3aug_show_geo(designs_circle,designs_poly,designs_points, ds = ['poly']*10):
    n = len(designs_poly)
    chiral = arm3aug_one_ligament_asym(*designs_circle, 
                          n = n,
                          ds = mix_space(ds))
    if chiral.valid_lig(designs_poly):
        chiral.plot_geo(designs_poly,designs_points.reshape(-1,2))

        
def get_ligs_arm3aug(config,circles,points):
    assert len(points) == len(circles)
    arm_config = config.arm_configs['arm3']
    ligs = []
    for i in range(len(points)):
        chiral = arm3aug_one_ligament_asym(R=circles[i][0],
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


def get_features_arm3aug(config,circles,ligs):
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
        x1 = -L/2+R*math.cos(theta1)
        y1 = R*math.sin(theta1)

        x00 = L/2+R*math.cos(theta00)
        y00 = R*math.sin(theta00)
        x11 = L/2+R*math.cos(theta11)
        y11 = R*math.sin(theta11)

        xss = L/2 + (R+2*r_s)*np.cos(theta00)
        yss = (R+2*r_s)*np.sin(theta00)

        x = list(np.linspace(x0,xss,100)) 
        y = [sumf(c=a,f = ds.f, x=i) for i in x]
        features.append(x + y + [x1,y1,x0,y0,x00,y00,x11,y11])
    features = np.array(features)
    #features = (features-arm_config.feature_mean[0])/arm_config.feature_std[0]
    return features


if __name__=='__main__':
    ds = ['poly']*7
    ligs, circles, points = arm3aug_asym_poly_designs(n_designs = 2, n=7, ds_name=ds)
    for i in range(2):
        arm3aug_show_geo(designs_circle= circles[i],designs_poly = ligs[i],designs_points = points[i],ds = ds)
        