#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:43:27 2019

@author: ocan
"""
ONCLUSTER=False  ## enable this only for cluster use

import numpy as np
import matplotlib
if ONCLUSTER:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os,sys
from moirev5 import generate,updiag
from chern import ChernNumber
from buildHBdG import diag_H_BdG
from kerreffect import sigma_H
import functions
import multiprocessing
import gc
import time

#################################################################
#################################################################
############ A V A I L A B L E     F U N C T I O N S ############
#################################################################
#################################################################
# Once self consistent calculation is run and we obtain the resulting
# superconducting order parameters (or preload them from an earlier
# calculation) the following functions are available:
# 
###### ComputeChern(M) 
# will compute the Chern number where M^2 is the number of k-points
# for that specific calculation.
# M > 200 recommended.
#
#
###### ComputeMinGap()
# computes the minimum energy excitation gap. This function follows
# a refining algorithm which zooms in to smaller and smaller regions
# of the BZ near the Dirac cones where the gap opens to get a more
# accurate value of the gap as we are using a discrete lattice.
#
###### PlotSpectrum(M,plottype)
# plots the lowest bands (near zero energy) in 2D or 3D plot.
# M is again the size of the BZ k-mesh and plottype should be 
# a string "2D" or "3D" accordingly.
#
# example: PlotSpectrum(200,"2D")
##################################################################
##################################################################
################ M O D E L    P A R A M E T E R S ################
##################################################################
##################################################################

Temperature=0.01 # Kelvins
vec=[1,2]  #[0,1] for no twist
t=.153 # NN hopping in plane
tprime=-0.45*t # NNN hopping in plane
mu=-1.3*t # chemical potential chosen near optimal doping
V=0.146 # attractive pairing potential, adjusted so that 40meV 
        # maximal gap is obtained when there is no interlayer coupling
g_0=0.02 # interlayer coupling energy scale
offsetlayers=0.0 # offsets the second layer from the rotation center
a=1.0 ##lattice spacing (Cu-O square lattice) - unit length
d=2.22 ##interlayer distance - in units of a
interlayer_closest_neighbors=300 #when generating the interlayer terms
#this parameter determines the number of closest neighbors to a lattice
#site. We pick a large number for this as the coupling strength (according
# to our model) decreases exponentially with distance.
max_interlayer_in_plane=7.8 # maximum range (in unit length a - latice spacing)
# for interlayer terms. 

##################################################################
##################################################################
######################### M E T H O D S ##########################
##################################################################
##################################################################

M=40 # specifies the BZ mesh - (M x M k-point grid) 

plot_op_iterations = True # visual for iterations and tracking convergence.
# it shows the value of each order parameter living in bonds between
# sites (blue: real part and red: imaginary part). For each iteration
# these values are plotted and when dots stop moving we know that 
# iterations have converged.
# code will also print a metric "diff" to show the difference between
# iterations, this should ideally be of the order 1e-10 or smaller.

saveresults=True  # saves self consistent calculation results in disk.
plot_hop_map=False # visualizes the hoppings in the tight binding model.
USEPREVCALCS=True # if results from earlier calculation has been saved, 
                  # code will load these insted of the ansatz - allows
                  # more iterations.
parallel= True    # k-sums in the gap equation can be done in parallel
localcpus=7       # specify the number of cores to be used for parallel calculation.
keeplayer1real=True # fixes the complex phase of order parameters along 
                    # the x direction in layer 1 - chooses a gauge.
mixing = 0.7      # mixing between the current and previous iteration to
                  # improve convergence. 
total_iterations=1000   # total number of iterations (recommended 1000)
numericalaccuracy = 1e-12 # stops the iterations if this is reached before
                  # doing "ti" iterations.

##################################################################
##################################################################

np.seterr(over="ignore")
os.environ["OMP_NUM_THREADS"] = "1"


if ONCLUSTER:
    CPUCOUNT= int(sys.argv[6])
else:
    CPUCOUNT=localcpus

kB=8.617e-5
ti=total_iterations
TBparameters = mu,t,tprime,g_0

if ONCLUSTER:
    g_0=np.round(float(sys.argv[1]),4)
    M=int(sys.argv[4])
    vec[0]=int(sys.argv[2])
    vec[1]=int(sys.argv[3])
    mu=np.round(float(sys.argv[7]),4)*t
    
offsetamount=offsetlayers  ## OFFSET - TRANSLATING THE TOP LAYER. VALUES BETWEEN 0-1
## ansatz phase
extrapolate=True
phi=np.pi*-0.4 #phase between two layers.
#phi=np.pi*-0.4 #phase between two layers.

deltamaxfactor=4 ### Delta_max = 4*Delta_x (or Delta_y)


if ONCLUSTER: ti=int(sys.argv[5])

### ANSATZ PARAMETERS ### 
Delta_ansatz=0.01 #.01 #Delta_x initial value. Delta_max = 0.04eV/4!

phase=np.exp(1.0j*phi)  ## relative phase between layers.
Delta_x1=1.0*Delta_ansatz
Delta_y1=-1.0*Delta_ansatz
Delta_x2=1.0*phase*Delta_ansatz
Delta_y2=-1.0*phase*Delta_ansatz

savename="twist"+str(vec[0])+"-"+str(vec[1])+"_g0_"+str(np.round(g_0,4))+"_mu_"+str(np.round(mu/t,4))+"t_M_"+str(M)


### GENERATE HOPPINGS
if 'intralayerhops' not in globals():
    print ("generating hopping terms..")
    hops1,hops2,hops12,hops11,hops22=generate(a,d,vec,interlayer_closest_neighbors,max_interlayer_in_plane,offsetamount,plot_hop_map)
    hops1=updiag(hops1) ##put all hoppings on the uppertriangular H
    hops2=updiag(hops2) ##same for other layer
    hops11=updiag(hops11) ## next nearest neighbors
    hops22=updiag(hops22) ## next nearest neighbors
    intralayerhops=hops1+hops2 #combine the two lists.

### SELF CONSISTENT MEAN FIELD TREATMENT
def init(T,g0,total_iterations,deltas="useansatz",vec=[0,1],M_mesh=50):
    global intralayerhops,hops12,hops11,hops22,savename,unitcellsize,basissize,oneDk,kx,ky,H0CREATED,Hcreatedonce

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
        if plot_op_iterations:
            oplist=np.array(intralayerhops)[:,2]
            plt.figure(1000)
            palpha=1-np.exp(-.1*iteration/total_iterations)
            plt.plot(deltas.real,'.',color="blue",alpha=palpha)
            plt.plot(deltas.imag,'.',color="red",alpha=palpha)
            plt.xticks(ticks=np.arange(len(oplist)),labels=oplist)
            
        if parallel: 
            def para_diag(index,returndict):
                dkvecs_slice=dkvecs[index[0]:index[1]+1]
                H=diag_H_BdG(dkvecs_slice,basissize,unitcellsize,intralayerhops,hops12,deltas,hops11,hops22,TBparameters)
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
    
            H=diag_H_BdG(dkvecs,basissize,unitcellsize,intralayerhops,hops12,deltas,hops11,hops22,TBparameters)
            
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
                pstring="%"+str(np.round(iteration/total_iterations*100,1))+"diff: "+str(realdiff)
                print (pstring)
                if saveresults:
                    file = open(savename+"/"+savename+".txt","a+") 
                    file.write(pstring+"\r")
                    file.close()
        if basissize >= 50:
            pstring="%"+str(np.round(iteration/total_iterations*100,1))+"diff: "+str(realdiff)
            print (pstring)
            if saveresults:
                file = open(savename+"/"+savename+".txt","a+") 
                file.write(pstring+"\r")
                file.close()
            
        iteration+=1
    
  
    finaldiff=np.sqrt(realdiff**2+imagdiff**2)    

    '''

    lb=evals[:,basissize//2]

    plt.plot(evals[:,basissize//2])
    '''
    '''
    H=diag_H_BdG(dkvecs,basissize,unitcellsize,intralayerhops,hops12,g0,deltas)
    evals=LA.eigvalsh(H,UPLO="U")
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
    if saveresults:
        np.save(savename+"/deltas",deltas)
        print ("saved order parameters in disk")
    return deltas,finaldiff


currentdeltas=[]
T=Temperature*kB
    

if not os.path.exists(savename):
    os.mkdir(savename)
    print("Directory " , savename ,  " Created ")
    print("using d-wave type ansatz")
    print ("Now running "+str(ti)+ " iterations:")
    currentdeltas,diff=init(T,total_iterations=ti,g0=g_0,M_mesh=M,deltas=currentdeltas,vec=vec)

else:
    if os.path.exists(savename+"/deltas.npy"):
        print ("Previous calculation has been found!")
        print("directory: /"+savename ,)
    if USEPREVCALCS:
        print ("loading the existing result for order parameters.")
        currentdeltas=np.load(savename+"/deltas.npy")
        runmore=input("Should we run more iterations? [Y/N]")
        if runmore in ["Y","y","yes","YES"]:
            print ("Now running "+str(ti)+ " iterations:")
            currentdeltas,diff=init(T,total_iterations=ti,g0=g_0,M_mesh=M,deltas=currentdeltas,vec=vec)
            
    else:
        print ("But d-wave type ansatz will be used instead. \nSet USEPREVCALCS=True in METHODS to load existing data")
        print ("Now running "+str(ti)+ " iterations:")
        currentdeltas,diff=init(T,total_iterations=ti,g0=g_0,M_mesh=M,deltas=currentdeltas,vec=vec)

           
def ComputeMinGap():
    mingap=functions.minimum_gap(M,CPUCOUNT,currentdeltas,diag_H_BdG,vec,intralayerhops,hops12,hops11,hops22,TBparameters)
    print ("Minimum gap : "+str(np.round(mingap*1000,5))+ "meV")

def ComputeChern(chern_BZsize):
    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2) ## size of the UC in terms of lattice spacing "a".
    basissize=4*(vec[0]**2+vec[1]**2) #2x2 block for each site - dimensions of the basis.
    oneDk=((np.arange(chern_BZsize)/chern_BZsize)-0.5)*2*np.pi/unitcellsize 
    kx, ky = np.meshgrid(oneDk,oneDk)
    kvecs=np.column_stack((kx.flatten(),ky.flatten())) ## combine kvecs into an array.
    dk=np.abs(oneDk[1]-oneDk[0])
    H=diag_H_BdG(kvecs,basissize,unitcellsize,intralayerhops,hops12,currentdeltas,hops11,hops22,TBparameters)
    chernnumbercomputed=ChernNumber(H,dk)
    print ("Chern Number : "+str(np.round(chernnumbercomputed,5)))
            
    
    
def PlotSpectrum(spectrum_BZsize,plottype):
    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2) ## size of the UC in terms of lattice spacing "a".
    basissize=4*(vec[0]**2+vec[1]**2) #2x2 block for each site - dimensions of the basis.
    oneDk=((np.arange(spectrum_BZsize)/spectrum_BZsize)-0.5)*2*np.pi/unitcellsize 
    kx, ky = np.meshgrid(oneDk,oneDk)
    kvecs=np.column_stack((kx.flatten(),ky.flatten())) ## combine kvecs into an array.
    H=diag_H_BdG(kvecs,basissize,unitcellsize,intralayerhops,hops12,currentdeltas,hops11,hops22,TBparameters)
    evals=LA.eigvalsh(H,UPLO="U")
    evalsplot=evals.reshape(spectrum_BZsize,spectrum_BZsize,basissize)
    #evalsplot=evals,reshape(BZsize,BZsize,basissize)
    ## contour plot of one of the lower bands.
    if plottype == "2D":
        plt.figure()
        cmap2 = plt.get_cmap('inferno')
        plt.pcolormesh(kx, ky, -1.0*evalsplot[:,:,basissize//2-1], cmap=cmap2)
        plt.colorbar()
    if plottype == "3D":
        from mpl_toolkits.mplot3d import Axes3D    
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        ha.plot_surface(kx, ky, evalsplot[:,:,basissize//2])  
        ha.plot_surface(kx, ky, evalsplot[:,:,basissize//2-1])     


def ComputeKerr(BZsize):
    freq=np.arange(0,1.0,0.01)
    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2) ## size of the UC in terms of lattice spacing "a".
    basissize=4*(vec[0]**2+vec[1]**2) #2x2 block for each site - dimensions of the basis.
    oneDk=((np.arange(BZsize)/BZsize)-0.5)*2*np.pi/unitcellsize 
    dk=np.abs(oneDk[1]-oneDk[0])
    kx, ky = np.meshgrid(oneDk,oneDk)
    kvecs=np.column_stack((kx.flatten(),ky.flatten())) ## combine kvecs into an array.
    ### normal part of the Hamiltonial - we set order parameters to zero.
    H0=diag_H_BdG(kvecs,basissize,unitcellsize,intralayerhops,hops12,currentdeltas*0.0,hops11,hops22,TBparameters)
    ### and the Hamiltonian
    H=diag_H_BdG(kvecs,basissize,unitcellsize,intralayerhops,hops12,currentdeltas,hops11,hops22,TBparameters)
    return freq,sigma_H(H0,H,dk,T,freq)
'''



delta1x,delta1y,delta2x,delta2y=functions.getdeltas(intralayerhops,currentdeltas)

np.save(savename+"/deltas",currentdeltas)

print (np.round(np.array([np.angle(delta1y),np.angle(delta2x),np.angle(delta2y),np.angle(delta2x)-np.angle(delta2y)]),4)/np.pi)
print (np.round(np.array([np.arctan(delta1y),np.arctan(delta2x.imag/delta2x.real),np.arctan(delta2y.imag/delta2y.real),np.arctan(delta2x.imag/delta2x.real)-np.arctan(delta2y.imag/delta2y.real)])/np.pi,4))

delta1x,delta1y,delta2x,delta2y=functions.getdeltas(intralayerhops,currentdeltas)

print (functions.getdeltas(intralayerhops,currentdeltas))



DATA={}
DATA["ChernNumber"]=BC
DATA["Ts"]=T
DATA["Deltas"]=currentdeltas
DATA["avgDeltas"]=[delta1x,delta1y,delta2x,delta2y]
DATA["diffs"]=diff
DATA["mingap"]=mingap
np.save(savename+"/"+savename,DATA)
'''

