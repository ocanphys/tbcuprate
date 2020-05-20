#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:08:07 2020

@author: ocan
"""
import numpy as np

def diag_H_BdG(kvec,basissize,unitcellsize,intralayerhops,hops12,OPs,hops11,hops22,TBparameters):
    ''' constructs the hamiltonian '''
    ''' takes k-vector array and list of Deltas for each hopping - with the order of hopping terms '''
    kx,ky=kvec[:,0],kvec[:,1]
    H=np.zeros((len(kvec),basissize,basissize), dtype="complex")
    mu,t,tprime,g0=TBparameters
    muterm=np.zeros((basissize))
    muterm[::2] = -1
    muterm[1::2] = 1
    
    muterm = np.diag(mu*muterm)
    H=np.array(H+muterm)
        
    for hop_index in range(len(intralayerhops)):
        hop=intralayerhops[hop_index]
        m,n=hop[0],hop[1]
        Gx=hop[3][0]*unitcellsize
        Gy=hop[3][1]*unitcellsize
        kdotG=kx*Gx+ky*Gy
        
        ## KINETIC TERM - we are constructing only the upper diagonal.
        t1=-1.0*t*np.exp(-1.0j*(kdotG))
        t2=1.0*t*np.exp(-1.0j*(kdotG))
        H[:,2*m,2*n]+=t1
        H[:,2*m+1,2*n+1]+=t2
        Delta_bond=OPs[hop_index]
        d1=Delta_bond*np.exp(-1.0j*(kdotG))
        d2=np.conjugate(Delta_bond)*np.exp(-1.0j*(kdotG))
        H[:,2*m,2*n+1]+=d1
        H[:,2*m+1,2*n]+=d2
        ### must add C.C. for [0,1] is special, Hamiltonian can not be constructed only on the upper diagonal.
        ### can comment them out for twisted case but let's keep them to be safe.
        H[:,2*n,2*m]+=np.conjugate(t1)
        H[:,2*n+1,2*m+1]+=np.conjugate(t2)
        H[:,2*n+1,2*m]+=np.conjugate(d1)
        H[:,2*n,2*m+1]+=np.conjugate(d2)


    ## INTERLAYER HOPPINGS - closest neighbour.
    for hop in np.array(hops12, dtype=object):
        m,n=hop[0],hop[1]
        Gx=hop[3][0]*unitcellsize
        Gy=hop[3][1]*unitcellsize
        kdotG=kx*Gx+ky*Gy
        g = g0*hop[2]
        H[:,2*m,2*n]+= g*np.exp(-1.0j*(kdotG))
        H[:,2*m+1,2*n+1]+= -1.0*g*np.exp(-1.0j*(kdotG)) 
        H[:,2*n,2*m]+= np.conjugate(g*np.exp(-1.0j*(kdotG)))
        H[:,2*n+1,2*m+1]+= np.conjugate(-1.0*g*np.exp(-1.0j*(kdotG)))
        
        
    ## NEXT NEAREST NEIGHBORS
    NNNhops=hops11+hops22
    for hop in np.array(NNNhops, dtype=object):
        m,n=hop[0],hop[1]
        Gx=hop[3][0]*unitcellsize
        Gy=hop[3][1]*unitcellsize
        kdotG=kx*Gx+ky*Gy
        t1=-1.0*tprime*np.exp(-1.0j*(kdotG))
        t2=1.0*tprime*np.exp(-1.0j*(kdotG))
        H[:,2*m,2*n]+=t1
        H[:,2*m+1,2*n+1]+=t2       
        H[:,2*n,2*m]+=np.conjugate(t1)
        H[:,2*n+1,2*m+1]+=np.conjugate(t2)

    return H
