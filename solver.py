#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:43:27 2019

@author: ocan
"""

ONCLUSTER=False
USEPREVCALCS=True

import numpy as np
import matplotlib
if ONCLUSTER:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os,sys
from moirev5 import generate,updiag
#from moirev4gs_NN import generate,updiag
import functions
import multiprocessing
import gc
import time

np.seterr(over="ignore")
os.environ["OMP_NUM_THREADS"] = "1"
if ONCLUSTER:
    CPUCOUNT= int(sys.argv[6])
else:
    CPUCOUNT=7
parallel= True

Temperature=0.01
vec=[1,2]  #[0,1] for no twist
M=40 #BZsize x BZsize k-mesh

t=.153
tprime=-0.45*t
mu=-1.3*t
V=0.146
g_0=0.04
kB=8.617e-5

t1=time.time()
if ONCLUSTER:
    g_0=np.round(float(sys.argv[1]),4)
    M=int(sys.argv[4])
    vec[0]=int(sys.argv[2])
    vec[1]=int(sys.argv[3])
    mu=np.round(float(sys.argv[7]),4)*t
offsetamount=0.0  ## OFFSET - TRANSLATING THE TOP LAYER. VALUES BETWEEN 0-1

## ansatz phase
extrapolate=True
phi=np.pi*0.1 #phase between two layers.
#phi=np.pi*-0.4 #phase between two layers.

a=1.0 ##lattice spacing
d=2.22 ##interlayer distance
interlayer_closest_neighbors=300 #pick something large - takes at least 10 closest neighbors. but we keep only ones within the max distance defined in the next line
max_interlayer_in_plane=7.8 #max in_plane_interlayer distance - in units of a - lattice spacing.
deltamaxfactor=4 ### Delta_max = 4*Delta_x (or Delta_y)
### generate hoppings here:
plot_hop_map=False

numericalaccuracy = 1e-20
keeplayer1real=True
mixing = 0.8
ti=30
if ONCLUSTER: ti=int(sys.argv[5])

### ANSATZ PARAMETERS ### 
Delta_ansatz=0.01 #.01 #Delta_x initial value. Delta_max = 0.04eV/4!

phase=np.exp(1.0j*phi)  ## relative phase between layers.
Delta_x1=1.0*Delta_ansatz
Delta_y1=-1.0*Delta_ansatz
Delta_x2=1.0*phase*Delta_ansatz
Delta_y2=-1.0*phase*Delta_ansatz

savename="twist"+str(vec[0])+"-"+str(vec[1])+"_g0_"+str(np.round(g_0,4))+"_mu_"+str(np.round(mu/t,4))+"t_M_"+str(M)


def diag_H_BdG(kvec,basissize,unitcellsize,intralayerhops,hops12,g0,OPs,hops11,hops22):
    ''' constructs the hamiltonian '''
    ''' if solve=False, returns H. Else, returns evals,P,Pdag such that PdagHP = evals'''
    ''' takes k-vector array and list of Deltas for each hopping - with the order of hopping terms '''
    kx,ky=kvec[:,0],kvec[:,1]
    H=np.zeros((len(kvec),basissize,basissize), dtype="complex")
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



 ### INITIALIZE THE BZ -

def init(T,g0,total_iterations,deltas="useansatz",vec=[0,1],M_mesh=50):
    global intralayerhops,hops12,hops11,hops22,savename,unitcellsize,basissize,oneDk,kx,ky,H0CREATED,Hcreatedonce
    ### GENERATE HOPPINGS
    hops1,hops2,hops12,hops11,hops22=generate(a,d,vec,interlayer_closest_neighbors,max_interlayer_in_plane,offsetamount,plot_hop_map)
    hops1=updiag(hops1) ##put all hoppings on the uppertriangular H
    hops2=updiag(hops2) ##same for other layer
    hops11=updiag(hops11) ## next nearest neighbors
    hops22=updiag(hops22) ## next nearest neighbors
    intralayerhops=hops1+hops2 #combine the two lists.

    ### INITIALIZE THE BZ -
    BZsize=M_mesh ## M x M BZ k-mesh
    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2) ## size of the UC in terms of lattice spacing "a".
    basissize=4*(vec[0]**2+vec[1]**2) #2x2 block for each site - dimensions of the basis.
    oneDk=((np.arange(BZsize)/BZsize)-0.5)*2*np.pi/unitcellsize ## put units of BZ from -pi/L to pi/L
    kx, ky = np.meshgrid(oneDk,oneDk)
    kvecs=np.column_stack((kx.flatten(),ky.flatten())) ## combine kvecs into an array.
    
    dkvecs=kvecs
    #### if there is no initial OPs given, initiate an ansatz:

    if len(deltas) != len(intralayerhops):
        deltas=np.zeros((len(intralayerhops)),dtype="complex") ### we will use this array for order parameters (OPs)
        for hop_index in range(len(intralayerhops)):
            hop=intralayerhops[hop_index]
            if hop[2] == "+x" and hop[4][0]==1: delta_hop = Delta_x1
            if hop[2] == "+x" and hop[4][0]==2: delta_hop = Delta_x2
            if hop[2] == "+y" and hop[4][0]==1: delta_hop = Delta_y1
            if hop[2] == "+y" and hop[4][0]==2: delta_hop = Delta_y2
            deltas[hop_index] = delta_hop
    
    #### now create the Hamiltonian with these OPs and diagonalize, using dkvecs!
    realdiff=1.0
    imagdiff=1.0
    
    iteration=0
    while iteration < total_iterations and max(realdiff,imagdiff)>numericalaccuracy :
        ## this part plots to check the order parameters as we iterate
        '''
        plt.figure(999)
        palpha=1-np.exp(-.1*iteration/total_iterations)
        plt.plot(deltas.real,'.',color="blue",alpha=palpha)
        plt.plot(deltas.imag,'.',color="red",alpha=palpha)
        '''
        
        if parallel: 
            def para_diag(index,returndict):
                dkvecs_slice=dkvecs[index[0]:index[1]+1]
                H=diag_H_BdG(dkvecs_slice,basissize,unitcellsize,intralayerhops,hops12,g_0,deltas,hops11,hops22)
                evals,evecs=LA.eigh(H,UPLO="U")
                P=evecs
                Pdag=np.conjugate(evecs.transpose(0,2,1))
                nFspectrum=functions.nF(evals,T)    
                #print (iteration,realdiff,imagdiff)
                
                def DeltaNN_from_PHP(hop,P,Pdag,nFspectrum):
                    ##NN order parameter-
                    ##computes Delta_ab = -V/2N*\sum_ke^{-ikG_ab}[P.n_F.P^\dag](k)_(2b,2a+1)
                    V_cpu=V*1.0/CPUCOUNT
                    m,n=hop[0],hop[1]
                    Pa=P[:,2*n,:]
                    Pdagb=Pdag[:,:,2*m+1]
                    kx,ky=dkvecs_slice[:,0],dkvecs_slice[:,1]
                    Gx=hop[3][0]*unitcellsize
                    Gy=hop[3][1]*unitcellsize
                    kdotG=kx*Gx+ky*Gy
                    #kdotG=np.einsum('kl,l->k',kvecs,[Gx,Gy])
                    f=np.exp(-1.0j*kdotG)
                    d1=-0.5*V_cpu*(1.0/len(kdotG))*np.einsum('kq,kq,kq,k',Pa,nFspectrum,Pdagb,f)
                    Pa=P[:,2*m,:]
                    Pdagb=Pdag[:,:,2*n+1]
                    f=np.exp(1.0j*kdotG)
                    d2=-0.5*V_cpu*(1.0/len(kdotG))*np.einsum('kq,kq,kq,k',Pa,nFspectrum,Pdagb,f)
                    
                    ### THIS PREFACTOR CHANGES FOR CHUNKING, MUST SEND V TO V/CPUS
                    res =(d1+d2)
                    if keeplayer1real:
                        if hop[4][0] == 1 and hop[2] == "+x":
                            res=res.real
                    return res 
                
                newdeltas_chunk=[]
                for hop in intralayerhops:
                    newdeltas_chunk.append(DeltaNN_from_PHP(hop,P,Pdag,nFspectrum))
                newdeltas_chunk=np.array(newdeltas_chunk)
                returndict[index[0]]=newdeltas_chunk
            
            def splitindices(chunknumber):
                index=np.arange(len(dkvecs))
                splitindex=np.array_split(index,chunknumber)
                slicelist=[]
                for section in splitindex:
                    slicelist.append([section[0],section[-1]])
                return slicelist


            cpus = CPUCOUNT
            #print ("using "+str(cpus)+" CPUS:")
            #t1=time.time()
            if __name__ == '__main__':
                manager = multiprocessing.Manager()
                #sharedmem_deltas = manager.list(np.zeros((len(intralayerhops)),dtype="complex"))
                returndict = manager.dict()
                processes=[]
                for indexslices in splitindices(cpus):
                    #print (indexslices)
                    p = multiprocessing.Process(target=para_diag,args=(indexslices,returndict))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()                
            

            newdeltas=np.zeros((len(intralayerhops)),dtype="complex")
            for delta_chunk in returndict.values():
                newdeltas += delta_chunk
            ### IMPORTANT!!
            #t2=time.time()
            #print ("diagonalize:"+str(t2-t1))
        else:
    
            H=diag_H_BdG(dkvecs,basissize,unitcellsize,intralayerhops,hops12,g0,deltas,hops11,hops22)
            
            evals,evecs=LA.eigh(H,UPLO="U")
            
            P=evecs
            Pdag=np.conjugate(evecs.transpose(0,2,1))
            nFspectrum=functions.nF(evals,T)    
            #print (iteration,realdiff,imagdiff)
            def DeltaNN_from_PHP(hop):
                ##NN order parameter-
                ##computes Delta_ab = -V/2N*\sum_ke^{-ikG_ab}[P.n_F.P^\dag](k)_(2b,2a+1)
                
                m,n=hop[0],hop[1]
                Pa=P[:,2*n,:]
                Pdagb=Pdag[:,:,2*m+1]
                kx,ky=dkvecs[:,0],dkvecs[:,1]
                Gx=hop[3][0]*unitcellsize
                Gy=hop[3][1]*unitcellsize
                kdotG=kx*Gx+ky*Gy
                #kdotG=np.einsum('kl,l->k',kvecs,[Gx,Gy])
                f=np.exp(-1.0j*kdotG)
                d1=-0.5*V*(1.0/len(kdotG))*np.einsum('kq,kq,kq,k',Pa,nFspectrum,Pdagb,f)
                Pa=P[:,2*m,:]
                Pdagb=Pdag[:,:,2*n+1]
                f=np.exp(1.0j*kdotG)
                d2=-0.5*V*(1.0/len(kdotG))*np.einsum('kq,kq,kq,k',Pa,nFspectrum,Pdagb,f)
                res =(d1+d2)
                if keeplayer1real:
                    if hop[4][0] == 1:
                        res=res.real
                return res 
            
            newdeltas=[]
            for hop in intralayerhops:
                newdeltas.append(DeltaNN_from_PHP(hop))
            newdeltas=np.array(newdeltas)

        diff=-1.0*deltas+0.000
        deltas = newdeltas*mixing + deltas*(1-mixing)
        diff = diff + deltas
        realdiff=np.sum(np.abs(diff.real))/len(diff)
        imagdiff=np.sum(np.abs(diff.imag))/len(diff)
        
        if basissize <50:
            if iteration %10 == 0:
                file = open(savename+"/"+savename+".txt","a+") 
                pstring="%"+str(np.round(iteration/total_iterations*100,1))+"diff: "+str(realdiff)
                print (pstring)
                file.write(pstring+"\r")
                file.close()
        if basissize >= 50:
            file = open(savename+"/"+savename+".txt","a+") 
            pstring="%"+str(np.round(iteration/total_iterations*100,1))+"diff: "+str(realdiff)
            print (pstring)
            file.write(pstring+"\r")
            file.close()
        
        iteration+=1
    
    '''
    H=diag_H_BdG(dkvecs,basissize,unitcellsize,intralayerhops,hops12,g0,deltas)
    evals=LA.eigvalsh(H,UPLO="U")
    lb=evals[:,basissize//2]
    '''
    finaldiff=np.sqrt(realdiff**2+imagdiff**2)    

    '''
    plt.figure()'Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c'
    plt.plot(evals[:,basissize//2])
    '''
    '''
    evalsplot=evals.reshape(M,M,basissize)
    #evalsplot=evals,reshape(BZsize,BZsize,basissize)
    ## contour plot of one of the lower bands.
    cmap2 = plt.get_cmap('inferno')
    plt.pcolormesh(kx, ky, -1.0*evalsplot[:,:,basissize//2], cmap=cmap2)
    
    from mpl_toolkits.mplot3d import Axes3D    
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    ha.plot_surface(kx, ky, evalsplot[:,:,basissize//2])  
    ha.plot_surface(kx, ky, evalsplot[:,:,basissize//2-1]) 
    '''
    
    return deltas,finaldiff


currentdeltas=[]

    

if not os.path.exists(savename):
    os.mkdir(savename)
    print("Directory " , savename ,  " Created ")
else:
    print("Directory " , savename ,  " already exists")
    if USEPREVCALCS:
        if os.path.exists(savename+"/deltas.npy"):
            print ("prev .deltas file found!")
            currentdeltas=np.load(savename+"/deltas.npy")
            
T=Temperature*kB

currentdeltas,diff=init(T,total_iterations=ti,g0=g_0,M_mesh=M,deltas=currentdeltas,vec=vec)
delta1x,delta1y,delta2x,delta2y=functions.getdeltas(intralayerhops,currentdeltas)

np.save(savename+"/deltas",currentdeltas)

print (np.round(np.array([np.angle(delta1y),np.angle(delta2x),np.angle(delta2y),np.angle(delta2x)-np.angle(delta2y)]),4)/np.pi)
print (np.round(np.array([np.arctan(delta1y),np.arctan(delta2x.imag/delta2x.real),np.arctan(delta2y.imag/delta2y.real),np.arctan(delta2x.imag/delta2x.real)-np.arctan(delta2y.imag/delta2y.real)])/np.pi,4))

delta1x,delta1y,delta2x,delta2y=functions.getdeltas(intralayerhops,currentdeltas)

print (functions.getdeltas(intralayerhops,currentdeltas))

t2=time.time()

print (t2-t1)
'''
DATA={}
DATA["ChernNumber"]=BC
DATA["Ts"]=T
DATA["Deltas"]=currentdeltas
DATA["avgDeltas"]=[delta1x,delta1y,delta2x,delta2y]
DATA["diffs"]=diff
DATA["mingap"]=mingap
np.save(savename+"/"+savename,DATA)
'''

