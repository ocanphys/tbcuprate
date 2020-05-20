#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:50:49 2019

@author: ocan
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from numpy import linalg as LA
import matplotlib
import sys,os
import multiprocessing
from itertools import cycle

'''
    evalsplot=evals.reshape(M_finer,M_finer,basissize)
    lowestband=evalsplot[:,:,basissize//2]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
    cmap2 = plt.get_cmap('inferno')
    plt.figure()
    plt.pcolormesh(kx2, ky2, lowestband, cmap=cmap2,norm=norm)
'''

def minimum_gap(M_finer,CPUCOUNT,deltas,diag_H_BdG,vec,intralayerhops,hops12,hops11,hops22,TBparameters):

    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2) ## size of the UC in terms of lattice spacing "a".
    basissize=4*(vec[0]**2+vec[1]**2) #2x2 block for each site - dimensions of the basis.

    kmin_x,kmin_y=0.0,0.0
    ### create the same mesh but centered at kmin - with fractionofBZ
    fractionofBZ=1.0
    iterations=0
    while iterations < 15: # and accuracy > 1e-18:
        oneDkx_finer=((np.arange(M_finer)/M_finer-0.5)*2*np.pi/unitcellsize)/fractionofBZ + kmin_x
        oneDky_finer=((np.arange(M_finer)/M_finer-0.5)*2*np.pi/unitcellsize)/fractionofBZ + kmin_y
        fractionofBZ = fractionofBZ*2
        kx_finer, ky_finer = np.meshgrid(oneDkx_finer,oneDky_finer)
        kvecs_finer=np.column_stack((kx_finer.flatten(),ky_finer.flatten())) ## combine kvecs into an array.

        def para_diag(index,returndict):
            dkvecs_slice=kvecs_finer[index[0]:index[1]+1]
            H=diag_H_BdG(dkvecs_slice,basissize,unitcellsize,intralayerhops,hops12,deltas,hops11,hops22,TBparameters)
            evals=LA.eigvalsh(H,UPLO="U")
            min_index=np.argmin(evals[:,basissize//2])
            min_value=evals[min_index,basissize//2]
            returndict[index[0]]=dkvecs_slice[min_index],min_value
                            
        def splitindices(chunknumber):
            index=np.arange(len(kvecs_finer))
            splitindex=np.array_split(index,chunknumber)
            slicelist=[]
            for section in splitindex:
                slicelist.append([section[0],section[-1]])
            return slicelist

        cpus = CPUCOUNT
        #print ("using "+str(cpus)+" CPUS:")
        #t1=time.time()
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
            
        for vals in returndict.values():
            print (vals)
        
        print (returndict)
        minimum=sorted(returndict.items(),  key=lambda kv: kv[1][1])[0]
        kmin_x,kmin_y=minimum[1][0]
        min_value=minimum[1][1]
        
        
        '''
        H_fine=diag_H_BdG(kvecs_finer,basissize,unitcellsize,intralayerhops,hops12,g_0,deltas)
        evals=LA.eigvalsh(H_fine,UPLO="U")
        min_index=np.argmin(evals[:,basissize//2])
        min_value_new=evals[min_index,basissize//2]
        dgap=min_value_new-min_value
        min_value=min_value_new
        accuracy=np.abs(dgap)/np.abs(min_value)
        kmin_x,kmin_y=kvecs_finer[min_index]
        '''
        
        print ("found min at "+str(np.round(kmin_x,3))+","+str(np.round(kmin_y,3))+" as "+str(np.round(min_value,10))+" min ext (half the gap)")
        iterations +=1
    gap=2*min_value
    return gap

def fermi_ks(kvecs,H0,debye_frequency,plotfs,unitcellsize):
    basissize=len(H0[0])
    evals=LA.eigvalsh(H0,UPLO="U")[:,int(basissize/2)]
    kvecs_debye=[]
    for i in range(len(evals)):
        val = evals[i]
        if val <= debye_frequency: kvecs_debye.append(kvecs[i])
    kvecs_debye=np.array(kvecs_debye)
    
    if plotfs:
        BZsize=int(np.sqrt(len(H0)))
        oneDk=((np.arange(BZsize)/BZsize)-0.5)*2*np.pi/unitcellsize
        kx, ky = np.meshgrid(oneDk,oneDk)        
        fig = plt.figure()     
        plt.title("$\omega_D=$"+str(debye_frequency)+"eV near Fermi surface")
        evalsplot=evals.reshape(BZsize,BZsize,1)
        ## contour plot of one of the lower bands.
        cmap2 = plt.get_cmap('inferno')
        plt.pcolormesh(kx, ky, -1.0*evalsplot[:,:,0], cmap=cmap2)
        plt.colorbar()
        #plt.matshow(evalsplot[:,:,0])
        for kpoint in kvecs_debye:
            plt.scatter(kpoint[0],kpoint[1],marker=".",color="C0")
        plt.show()
    return kvecs_debye

def nF(energy,T):
    return 1.0/(np.exp(energy/T)+1.0)


def plotalldeltas(intralayerhops,Ds,diffs,Ts,mingaps,g_0,figname):
    fig, axs = plt.subplots(4,2,figsize=(10, 12))
    plt.subplots_adjust(hspace=0.3)
    finalplotalpha=0.8
    delta_2_x_real=0
    delta_2_x_imag=0
    delta_2_y_real=0
    delta_2_y_imag=0
    delta_1_x_real=0
    delta_1_x_imag=0
    delta_1_y_real=0
    delta_1_y_imag=0
    delta2x_count=0
    delta2y_count=0
    delta1x_count=0
    delta1y_count=0
    for hopindex in range(len(intralayerhops)):
        hop=intralayerhops[hopindex]
        delta_bond=Ds[:,hopindex]
        if hop[2] == "+x" and hop[4][0]==1: 
            axs[0,0].plot(Ts,delta_bond.real,color="blue",alpha=finalplotalpha)
            axs[0,0].plot(Ts,delta_bond.imag,color="red",alpha=finalplotalpha)
            delta_1_x_real+=delta_bond.real
            delta_1_x_imag+=delta_bond.imag
            delta1x_count+=1
        if hop[2] == "+y" and hop[4][0]==1:
            axs[1,0].plot(Ts,delta_bond.real,color="blue",alpha=finalplotalpha)
            axs[1,0].plot(Ts,delta_bond.imag,color="red",alpha=finalplotalpha)
            delta_1_y_real+=delta_bond.real
            delta_1_y_imag+=delta_bond.imag
            delta1y_count+=1
        if hop[2] == "+x" and hop[4][0]==2: 
            axs[2,0].plot(Ts,delta_bond.real,color="blue",alpha=finalplotalpha)
            axs[2,0].plot(Ts,delta_bond.imag,color="red",alpha=finalplotalpha)
            delta_2_x_real+=delta_bond.real
            delta_2_x_imag+=delta_bond.imag
            delta2x_count+=1
        if hop[2] == "+y" and hop[4][0]==2:
            axs[3,0].plot(Ts,delta_bond.real,color="blue",alpha=finalplotalpha)
            axs[3,0].plot(Ts,delta_bond.imag,color="red",alpha=finalplotalpha)
            delta_2_y_real+=delta_bond.real
            delta_2_y_imag+=delta_bond.imag
            delta2y_count+=1
    delta2x_real=delta_2_x_real/delta2x_count
    delta2x_imag=delta_2_x_imag/delta2x_count
    delta2y_real=delta_2_y_real/delta2y_count
    delta2y_imag=delta_2_y_imag/delta2y_count
    delta1x_real=delta_1_x_real/delta1x_count
    delta1x_imag=delta_1_x_imag/delta1x_count
    delta1y_real=delta_1_y_real/delta1y_count
    delta1y_imag=delta_1_y_imag/delta1y_count
    
    delta1x=delta1x_real+1.0j*delta1x_imag
    delta1y=delta1y_real+1.0j*delta1y_imag
    delta2x=delta2x_real+1.0j*delta2x_imag
    delta2y=delta2y_real+1.0j*delta2y_imag
    DELTA1=2*np.abs(delta1x-delta1y)
    DELTA2=2*np.abs(delta2x-delta2y)
    axs[0,1].plot(Ts,DELTA1,color="green",alpha=finalplotalpha,label="$\Delta^1$")
    axs[0,1].plot(Ts,DELTA2,color="orange",alpha=finalplotalpha,linestyle="dashed",label="$\Delta^2$")
    axs[0,1].legend()
    axs[0,1].set_ylim([0, 0.06])
    phase2x=np.arctan(delta2x_imag/delta2x_real)
    phase2y=np.arctan(delta2y_imag/delta2y_real)
    phase1y=np.arctan(delta1y_imag/delta1y_real)
    axs[2,1].plot(Ts,phase1y/np.pi,label="$\phi^1_y$")
    axs[2,1].plot(Ts,phase2x/np.pi,label="$\phi^2_x$")
    axs[2,1].plot(Ts,phase2y/np.pi,linestyle="dashed",label="$\phi^2_y$")
    axs[1,1].plot(Ts,mingaps)
    axs[3,1].semilogy(Ts,diffs,'*')
    axs[0,0].set_title(figname)
    axs[0,0].set_ylabel("$\Delta_x^1$")
    axs[0,0].set_xlabel("T")
    axs[1,0].set_ylabel("$\Delta_y^1$")
    axs[1,0].set_xlabel("T")
    axs[2,0].set_ylabel("$\Delta_x^2$")
    axs[2,0].set_xlabel("T")
    axs[3,0].set_ylabel("$\Delta_y^2$")
    axs[3,0].set_xlabel("T")
    axs[1,1].set_ylabel("$\Delta_{min}$")
    axs[2,1].set_ylabel("$\phi/\pi$")
    axs[3,1].set_ylabel("$\delta\Delta$")
    plt.style.use("ggplot")
    axs[2,1].legend()
    axs[0,0].set_ylim([-0.018, 0.018])
    axs[1,0].set_ylim([-0.018, 0.018])
    axs[2,0].set_ylim([-0.018, 0.018])
    axs[3,0].set_ylim([-0.018, 0.018])
    axs[1,1].set_ylim([0, 0.01])
    axs[2,1].set_ylim([-1.0, 1.0])
    plt.tight_layout()
    if not os.path.exists(figname):
        os.mkdir(figname)
        print("Directory " , figname ,  " Created ")
    else:
        print("Directory " , figname ,  " already exists")

    fig.savefig(str(figname)+"/"+str(figname)+".png")
    plt.close()
    
    
def getdeltas(intralayerhops,Ds):
    delta_2_x_real=0
    delta_2_x_imag=0
    delta_2_y_real=0
    delta_2_y_imag=0
    delta_1_x_real=0
    delta_1_x_imag=0
    delta_1_y_real=0
    delta_1_y_imag=0
    delta2x_count=0
    delta2y_count=0
    delta1x_count=0
    delta1y_count=0
    for hopindex in range(len(intralayerhops)):
        hop=intralayerhops[hopindex]
        delta_bond=Ds[hopindex]
        if hop[2] == "+x" and hop[4][0]==1: 
            delta_1_x_real+=delta_bond.real
            delta_1_x_imag+=delta_bond.imag
            delta1x_count+=1
        if hop[2] == "+y" and hop[4][0]==1:
            delta_1_y_real+=delta_bond.real
            delta_1_y_imag+=delta_bond.imag
            delta1y_count+=1
        if hop[2] == "+x" and hop[4][0]==2: 
            delta_2_x_real+=delta_bond.real
            delta_2_x_imag+=delta_bond.imag
            delta2x_count+=1
        if hop[2] == "+y" and hop[4][0]==2:
            delta_2_y_real+=delta_bond.real
            delta_2_y_imag+=delta_bond.imag
            delta2y_count+=1
    delta2x_real=delta_2_x_real/delta2x_count
    delta2x_imag=delta_2_x_imag/delta2x_count
    delta2y_real=delta_2_y_real/delta2y_count
    delta2y_imag=delta_2_y_imag/delta2y_count
    delta1x_real=delta_1_x_real/delta1x_count
    delta1x_imag=delta_1_x_imag/delta1x_count
    delta1y_real=delta_1_y_real/delta1y_count
    delta1y_imag=delta_1_y_imag/delta1y_count
    
    delta1x=delta1x_real+1.0j*delta1x_imag
    delta1y=delta1y_real+1.0j*delta1y_imag
    delta2x=delta2x_real+1.0j*delta2x_imag
    delta2y=delta2y_real+1.0j*delta2y_imag
    
    return delta1x,delta1y,delta2x,delta2y

def homvals(dhops):
    '''
    normalizes the free parameters in minimization
    '''
    layer1x=0
    l1xc=0
    layer1y=0
    l1yc=0
    layer2x=0
    l2xc=0
    layer2y=0
    l2yc=0
    for hop in dhops:
        if hop[2] == "+x" and hop[4]==1: 
            layer1x +=hop[5]
            l1xc+=1
        if hop[2] == "+y" and hop[4]==1:
            layer1y +=hop[5]
            l1yc+=1
        if hop[2] == "+x" and hop[4]==2: 
            layer2x +=hop[5]
            l2xc+=1
        if hop[2] == "+y" and hop[4]==2:
            layer2y +=hop[5]
            l2yc+=1
    layer1x=layer1x/l1xc
    layer1y=layer1y/l1yc
    layer2x=layer2x/l2xc
    layer2y=layer2y/l2yc
    normhops=[]
    for hop in dhops:
        if hop[4] == 1: 
            if hop[2] == '+x': normhops.append(layer1x)
            if hop[2] == '+y': normhops.append(layer1y)
        if hop[4] == 2: 
            if hop[2] == '+x': normhops.append(layer2x)
            if hop[2] == '+y': normhops.append(layer2y)
    return np.array(normhops)
    

def gethomoindices(dhops):
    '''
    normalizes the free parameters in minimization
    '''
    layer1x=[]
    layer1y=[]
    layer2x=[]
    layer2y=[]
    i=0
    for hop in dhops:
        if hop[2] == "+x" and hop[4]==1: 
            layer1x.append(i)
        if hop[2] == "+y" and hop[4]==1:
            layer1y.append(i)
        if hop[2] == "+x" and hop[4]==2: 
            layer2x.append(i)
        if hop[2] == "+y" and hop[4]==2:
            layer2y.append(i)
        i +=1
    return layer1x[0],layer1y[0],layer2x[0],layer2y[0]
    

def plotphase(hops):
    layer1x,layer1y,layer2x,layer2y=renormalize(hops)
    
    plt.figure()
    plt.arrow(0,0,layer1x.real,layer1x.imag,width=0.001,length_includes_head=True,alpha=0.3,color="red")
    plt.text(layer1x.real,layer1x.imag,"$\Delta_1x$")
    plt.arrow(0,0,layer1y.real,layer1y.imag,width=0.001,length_includes_head=True,alpha=0.3,color="blue")
    plt.text(layer1y.real,layer1y.imag,"$\Delta_1y$")
    plt.arrow(0,0,layer2x.real,layer2x.imag,width=0.001,length_includes_head=True,alpha=0.3,color="green")
    plt.text(layer2x.real,layer2x.imag,"$\Delta_2x$")
    plt.arrow(0,0,layer2y.real,layer2y.imag,width=0.001,length_includes_head=True,alpha=0.3,color="purple")
    plt.text(layer2y.real,layer2y.imag,"$\Delta_2y$")
    plt.xlim(-0.05,0.05)
    plt.ylim(-0.05,0.05)
    plt.legend()
    


####
    
'''

    
    cmap2 = plt.get_cmap('inferno')
    plt.figure()
    plt.pcolormesh(kx, ky, evalsplot[:,:,int(basissize/2+1)], cmap=cmap2)
    #plt.pcolormesh(kx, ky, evalsplot[:,:,int(basissize/2)], cmap=cmap2)
    plt.colorbar()
    plt.show()
    
    
    ## plot the 3D band structure     
    from mpl_toolkits.mplot3d import Axes3D    
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    #for b in range(basissize//2,basissize//2+1):
    for b in range(basissize//2,basissize//2+10):
        ha.plot_surface(kx, ky, evalsplot[:,:,b])  
    
    ## don't have to do this every time - this takes time - CAN BE OPTIMIZED.
'''
