import numpy as np
import matplotlib.pyplot as plt

from chern import ChernNumber
from numpy import linalg as LA
from kerreffect import sigma_H
#def twoband_H(kvec,basissize,unitcellsize,intralayerhops,hops12,OPs,hops11,hops22,TBparameters):

def twoband_H_BdG(kvec,vec,Delta_0,phase,TBparameters):
    kx,ky=kvec[:,0],kvec[:,1]
    angle_rad=2*np.arctan(vec[0]/vec[1])
    H=np.zeros((len(kvec),4,4), dtype="complex")    
    a=1.0
    b=np.sqrt(vec[0]**2+vec[1]**2)*a
    
    mu,t,tprime,g0=TBparameters
    mu=mu+4*t*(1-a**2/b**2)
    t=t*a**2/b**2
    tprime=0.5*t*a**4/b**4
    delta=Delta_0*a**2/b**2
    
    print ("effective mu = "+str(mu))
    print ("effective mu/t = "+str(mu/t))
    print ("effective t = "+str(t))
    print ("effective t' = "+str(tprime))
    print ("effective delta' = "+str(delta))
    
    dxy=np.sin(kx*b)*np.sin(ky*b)
    dx2y2=np.cos(kx*b)-np.cos(ky*b)
    theta = angle_rad
    
    e_twoband_1= -mu -2.0*t*(np.cos(kx*b)+np.cos(ky*b)) + tprime*(dxy*np.cos(theta)+dx2y2*np.sin(theta))
    delta_twoband_1 = 0.5*delta*(dx2y2*np.cos(theta)-dxy*np.sin(theta))
    theta = -1.0*angle_rad
    e_twoband_2= -mu -2.0*t*(np.cos(kx*b)+np.cos(ky*b)) + tprime*(dxy*np.cos(theta)+dx2y2*np.sin(theta))
    delta_twoband_2 = 0.5*delta*phase*(dx2y2*np.cos(theta)-dxy*np.sin(theta))
    
    H=np.zeros((len(kvec),4,4), dtype="complex")
    e_k_top = e_twoband_1
    e_k_bot = e_twoband_2
    
    delta_top = delta_twoband_1
    delta_bot = delta_twoband_2
    
    ###2*(Delta_x1*np.cos(kx)+Delta_y1*np.cos(ky))
    H[:,0,0]=e_k_top
    H[:,1,1]=-1.0*e_k_top
    H[:,2,2]=e_k_bot
    H[:,3,3]=-1.0*e_k_bot

    ## ADD PAIRINGS
    H[:,0,1]=delta_top
    H[:,1,0]=np.conjugate(delta_top)
    
    H[:,2,3]=delta_bot
    H[:,3,2]=np.conjugate(delta_bot)
    
    ## COUPLE TWO LAYERS
    H[:,0,2]+=g_0
    H[:,1,3]+=-1.0*g_0
    H[:,2,0]+=g_0
    H[:,3,1]+=-1.0*g_0
    
    return H













