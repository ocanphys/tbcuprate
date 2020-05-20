#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:17:31 2020

@author: ocan
"""

import itertools
import numpy as np
import gc
from numpy import linalg as LA

def gradientofH(H,dk):
    basissize=np.shape(H)[1]
    lenk=np.shape(H)[0]
    meshsize=int(np.sqrt(lenk))
    Hsq=np.reshape(H,(meshsize,meshsize,basissize,basissize))
    gradHx=np.gradient(Hsq,dk,axis=1)
    H_grad_x=np.reshape(gradHx,(lenk,basissize,basissize)) ##grad_x
    del gradHx
    gradHy=np.gradient(Hsq,dk,axis=0)
    H_grad_y=np.reshape(gradHy,(lenk,basissize,basissize)) ##grad_y
    del gradHy
    gc.collect()
    print ("gradient is done.")
    return H_grad_x,H_grad_y



def ChernNumber(H,dk):
    H_grad_x,H_grad_y=gradientofH(H,dk)
    evals,evecs=LA.eigh(H,UPLO="U")
    basissize=np.shape(H)[1]
    del H
    gc.collect()
    
    def splitindices(chunknumber):
        index=np.arange(len(evals))
        splitindex=np.array_split(index,chunknumber)
        slicelist=[]
        for section in splitindex:
            slicelist.append([section[0],section[-1]])
        return slicelist

    def computechunk(index):
        slice_evals=evals[index[0]:index[1]+1]
        slice_evecs=evecs[index[0]:index[1]+1]
        slice_H_grad_x=H_grad_x[index[0]:index[1]+1]
        slice_H_grad_y=H_grad_y[index[0]:index[1]+1]
        Hkx=np.einsum('kim,kij,kjn->kmn',np.conjugate(slice_evecs),slice_H_grad_x,slice_evecs,optimize=True)
        Hky=np.einsum('kim,kij,kjn->kmn',np.conjugate(slice_evecs),slice_H_grad_y,slice_evecs,optimize=True)
    
        #print ("Hkx,Hky done")

        EmEn=np.zeros((len(slice_evecs),basissize,basissize))
        perms = itertools.product(np.arange(0,basissize),np.arange(0,basissize))
        #print ("perms now")
        for perm in perms:
            m,n=perm
            if m!=n:
                EmEn[:,m,n]=1.0/(slice_evals[:,m]-slice_evals[:,n])**2
            else:
                EmEn[:,m,n]=0.0
        curvature= -1.0*(dk**2)*(np.einsum('knm,kmn,kmn->kn',Hkx,Hky,EmEn)-np.einsum('knm,kmn,kmn->kn',Hky,Hkx,EmEn)).imag
        Bcurvature=0
        for i in range(0,int(basissize//2)):
            Bcurvature += curvature[:,i]
    
        tosum=sum(Bcurvature/(2*np.pi))
        return tosum
    
    chunklist=splitindices(200)
    BC=0.0
    for index in chunklist:
        BC+=computechunk(index)

    return BC
