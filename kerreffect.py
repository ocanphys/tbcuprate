#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:50:05 2020

@author: ocan
"""

import numpy as np
from numpy import linalg as LA
from functions import gradientofH,nF
import itertools 



def compute(omega,T,u_x,u_y,evals,epsilon):
    #epsilon = 0.01 ## small epsilon.
    basissize=np.shape(evals)[1]
    nFspectrum=nF(evals,T) 
    perms = itertools.product(np.arange(0,basissize),np.arange(0,basissize))
    fab_fba=np.zeros((len(evals),basissize,basissize),dtype="complex")
    for perm in perms:
        a,b=perm
        #f_ab[:,a,b]=(nFspectrum[:,a]-nFspectrum[:,b])/((omega + 1.0j*epsilon)**2-1.0*(evals[:,a]-evals[:,b])**2)
        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################
        def f(a,b,omega):
            omegaE=omega + evals[:,a]-evals[:,b]
            return (nFspectrum[:,a]-nFspectrum[:,b])*(omegaE-1.0j*epsilon)/(omegaE**2+epsilon**2)
        fab_fba[:,a,b]=f(a,b,omega)-f(b,a,omega)
        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################
    ## sigma_H=ie^2\sum_k (absum)
    absum=sum(np.einsum('kab,kba,kab->k',u_x,u_y,fab_fba))
    return 1.0j*absum/(2*omega) #in units of e^2
    
def sigma_H(H0,H,dk,T,freq,epsilon):
    global mingap
    
    v_x,v_y=gradientofH(H0,dk) ## gradient of the normal part
    evals,evecs=LA.eigh(H,UPLO="U")          
    P=evecs
    Pdag=np.conjugate(evecs.transpose(0,2,1))
    basissize=np.shape(H)[1]
    ## Hkx=np.einsum('kim,kij,kjn->kmn',np.conjugate(slice_evecs),slice_H_grad_x,slice_evecs,optimize=True)
    u_x= np.einsum('kab,kbc,kcd->kad',Pdag,v_x,P,optimize=True)
    u_y= np.einsum('kab,kbc,kcd->kad',Pdag,v_y,P,optimize=True)
    conductance=[]
    total_frequency_number=len(freq)*1.0
    i=0

    for omega in freq:
        print (str((i*1.0/total_frequency_number)*100)+"%")
        res=compute(omega,T,u_x,u_y,evals,epsilon)
        conductance.append(res)
        i+=1
    
        
    min_index=np.argmin(evals[:,basissize//2])
    min_value=evals[min_index,basissize//2]
    print ("minimum excitation energy:" +str(min_value)+"eV")
    mingap=min_value
    return np.array(conductance),min_value


