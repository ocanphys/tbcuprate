#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:08:51 2020

@author: ocan
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import spatial


def updiag(hops):
    for hop in hops:
        hopfrom=hop[0]
        hopto=hop[1]
        if hopfrom > hopto:
            hop[0]=hopto
            hop[1]=hopfrom
            hop[3][0]=-1*hop[3][0]
            hop[3][1]=-1*hop[3][1]
    return hops

def sqlattice(N,M):
    return np.array([x for x in itertools.product(np.arange(-1*N,N), np.arange(-1*M,M))])

def translate(lattice,translation_x,translation_y):
    translation=np.array([translation_x,translation_y])
    lattice=lattice+translation
    return lattice

def rotate(lattice,theta):
    '''
    rotates a 2D lattice (array) CCW by angle \theta.
    '''
    rot= np.matrix([[np.cos(theta),-1.0*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.einsum("ij,aj->ai",rot,lattice)

def plotlat(fig,lt,col,alph=1):
    '''
    plots the lattice
    '''
    x,y = lt.T
    fig.scatter(x,y,alpha=alph,color=col)#,s=10)

def generate(a,d,vec,interlayerNNnumber,interlayerNNdistance,offsetamount=0.0,plothoppingmaps=True,rho=0.39,nx=1,ny=1):
    ucboundaryshift=a*0.1
    
    global dict1,dict2,allhops
    #nx,ny=1,1 ## stack primitive unit cell: 1-1 no stacking.
    #vec=[3,8]
    #offsetamount=0.0
    #a=1.0
    #interlayerNNnumber=300
    #interlayerNNdistance=3.2
    #d=2.22
    #rho=0.39
    
    v1,v2=vec
    
    plothops=False ## plots the hopping map as they are generated. (random numbering)
    
    lat=sqlattice(30,30)
    
    ## HERE WE DEFINE THE UNITCELL AND CREATE THE LAYERS.
    unitcellsize=np.sqrt(vec[0]**2+vec[1]**2)
    layer1=rotate(lat,np.arctan(vec[0]/vec[1])) #blue - to the left
    layer2=rotate(lat,-1.0*np.arctan(vec[0]/vec[1])) #orange to the right
    ### translate layer 2
    offsetangle=np.pi/4-np.arctan(vec[0]/vec[1])
    layer2=translate(layer2,offsetamount*a*0.5*np.sqrt(2)*np.cos(offsetangle),offsetamount*a*0.5*np.sqrt(2)*np.sin(offsetangle))
    
    layer1=layer1+np.array([ucboundaryshift,ucboundaryshift])
    layer2=layer2+np.array([ucboundaryshift,ucboundaryshift])
    
    
    #layer1=np.round(layer1,2)
    #layer2=np.round(layer2,10)
    
    largeruclimits=10.0
    
    unitlayer1=[]
    unitlayer2=[]
    largeruc1=[]
    largeruc2=[]
    
    uc_size_x=nx*unitcellsize
    uc_size_y=ny*unitcellsize
    
    '''
    def inunit(x,y):
        if a*-0.01 <= x < uc_size_x-a*0.01 and a*-0.01 <= y < uc_size_y-a*0.01: return True
        else: return False
    '''
    def inunit(x,y):
        if 0 <= x < uc_size_x and 0 <= y < uc_size_y: return True
        else: return False
    
    
    def bonddirection(layer,dx,dy):
        if layer == 1:
            if dx < 0 and dy>0: return "+y"
            if dx < 0 and dy<0: return "-x"
            if dx > 0 and dy<0: return "-y"
            if dx > 0 and dy>0: return "+x"
        if layer == 2:
            if dx > 0 and dy>0: return "+y"
            if dx < 0 and dy>0: return "-x"
            if dx < 0 and dy<0: return "-y"
            if dx > 0 and dy<0: return "+x"
            
    for pt in layer1:
        if inunit(pt[0],pt[1]):
            unitlayer1.append(pt)
        if -1.0*largeruclimits*a <= pt[0] < uc_size_x + a*largeruclimits and -1.0*largeruclimits*a <= pt[1]< uc_size_y + a*largeruclimits:
            largeruc1.append(pt)
    for pt in layer2:
        if inunit(pt[0],pt[1]):
            unitlayer2.append(pt)
        if -1.0*largeruclimits*a <= pt[0] < uc_size_x + a*largeruclimits and -1.0*largeruclimits*a <= pt[1]< uc_size_y + a*largeruclimits:
            largeruc2.append(pt)
    
    unitlayer1=np.array(unitlayer1)
    unitlayer2=np.array(unitlayer2)
    largeruc1=np.array(largeruc1)
    largeruc2=np.array(largeruc2)
    
    def plotbase(fig):
        ### PLOT UNITCELL -
        ax = fig.add_subplot(1, 1, 1,adjustable='box', aspect=1)
        #plt.xlim(0,uc_size_x)
        #plt.ylim(0,uc_size_y)
        plotlat(ax,largeruc1,"C0",alph=0.4)
        plotlat(ax,largeruc2,"C1",alph=0.4)
        plotlat(ax,unitlayer1,"C0")
        plotlat(ax,unitlayer2,"C1")
        rect = plt.Rectangle((0,0), uc_size_x, uc_size_y, color='k', alpha=0.1)
        ax.add_patch(rect)
    
    largeructree1=spatial.cKDTree(largeruc1)
    largeructree2=spatial.cKDTree(largeruc2)
    
    def get_hoppings(layer,neighbor_range=1):
        ## THIS CREATES THE NEAREST NEIGHBOR HOPPINGS FOR A GIVEN LAYER.
        hops=[]
        plothops=False
        if layer == 1: 
            largeruc=largeruc1
            largeructree = largeructree1
        if layer == 2 or layer ==12: 
            largeruc=largeruc2
            largeructree = largeructree2
        if plothops:
            fig=plt.figure()
            plotbase(fig)
        if layer ==1 or layer ==2:
            if neighbor_range == 1:
                NNhoppings=largeructree.query_pairs(1.05)
                layer_hoppings=NNhoppings
            if neighbor_range == 2:
                NN_NNNhoppings=largeructree.query_pairs(1.05*np.sqrt(2))
                NNhoppings=largeructree.query_pairs(1.05)
                difference=NN_NNNhoppings-NNhoppings
                layer_hoppings=difference
            if neighbor_range != 1 and neighbor_range != 2:
                print ("range not available.")
                return 
            for hop in layer_hoppings:
                depx,depy=largeruc[hop[0]]
                desx,desy=largeruc[hop[1]]
                dx=desx-depx
                dy=desy-depy
                dep_in=inunit(depx,depy)
                des_in=inunit(desx,desy)
                if dep_in:
                    bdirection=bonddirection(layer,dx,dy)
                    reciprocal=[int(desx//(uc_size_x)),int(desy//(uc_size_y))]
                    if plothops:
                        plt.text(depx+0.1*a,depy+0.1*a,hop[0]) #plot the old labeling
                        if des_in: plt.text(desx+0.1*a,desy+0.1*a,hop[1]) #plot the old labeling
                    
                        plt.arrow(depx,depy,dx,dy,width=0.02,length_includes_head=True,alpha=0.5)
                        plt.text(depx+0.5*dx,depy+0.5*dy,bdirection,color='blue')
                    
                    hops.append([hop[0],hop[1],bdirection,reciprocal,[layer,layer]])
                    #print ([hop[0],hop[1],bdirection,reciprocal,[layer,layer]])
        if layer == 12:        
            def findNNBS(siteindex,closest_n,maxdistance):
                largeructree=spatial.cKDTree(largeruc2)
                hopfrom=largeruc1[siteindex]
                NNBS=np.array(largeructree.query(hopfrom,closest_n)).T
                if closest_n == 1:
                    NNBS=np.array([NNBS])
                resNNBS=[]
                for interhop in NNBS:
                    if interhop[0] <= maxdistance*a:
                        resNNBS.append(interhop)
                return np.array(resNNBS)
            
            for siteindex in range(len(largeruc1)):
                hopfrom=largeruc1[siteindex]
                NNBS=findNNBS(siteindex,closest_n=interlayerNNnumber,maxdistance=interlayerNNdistance)
                for nneighbor in NNBS:
                    projecteddistance=nneighbor[0]
                    targetsite=int(nneighbor[1])
                    depx,depy=hopfrom
                    desx,desy=largeruc2[targetsite]
                    dx=desx-depx
                    dy=desy-depy
                    projecteddistance=np.sqrt(dx**2+dy**2)
                    distance=np.sqrt(projecteddistance**2 + d**2)

                    #couplingfactor=np.exp(rho/d)*np.exp(-distance/rho)
                    couplingfactor=np.exp(-1.0*(distance-d)/rho)

            
                    palpha=0.1
                    #reciprocal=[int(desx//(unitcellsize*0.999)),int(desy//(unitcellsize*0.999))]
                    reciprocal=[int(desx//(uc_size_x)),int(desy//(uc_size_y))]
                    if inunit(depx,depy):
                        palpha=0.4
                        if plothops:
                            plt.text(depx+0.05*a,depy+0.05*a,siteindex) #plot the old labeling
                            plt.text(desx+0.05*a,desy-0.05*a,targetsite) #plot the old labeling
                        hops.append([siteindex,targetsite,couplingfactor,reciprocal,[1,2]])
                        #print ("l1 site "+str(siteindex)+" to l2 site "+str(targetsite), reciprocal)
                        if plothops: plt.arrow(depx,depy,dx,dy,width=0.02*couplingfactor,length_includes_head=True,alpha=palpha) 
             
        newhops=[]
        i=0
        for hop in hops:
            rec=hop[3]
            target=hop[1]
            if rec != [0,0]:
                des=hop[1] #this is sticking out of the BZ.
                subtract=np.array([rec[0]*uc_size_x,rec[1]*uc_size_y])
                intheBZ=largeruc[des]-subtract # get the coordinates of this point and subtract the reciprocal vector 
                #intheBZ=largeruc[des]-np.array(rec)*unitcellsize # get the coordinates of this point and subtract the reciprocal vector 
                #now find the closest point IN the unit cell
                if not inunit(intheBZ[0],intheBZ[1]):
                    print (largeruc[des],"NOT BACK IN THE BZ!", intheBZ)
                    print (hop)
                target=largeructree.query(intheBZ,1)[1]
                #print (hop[0],hop[1]," goes back to",hop[0],target) 
                ## modify the hopping accordingly: np.array([hop[0],hop[1],bdirection,reciprocal,layer])
            #print ([hop[0],target,hop[2],hop[3],hop[4]])
            newhops.append([hop[0],target,hop[2],hop[3],hop[4]])
            i+=1
        return newhops
    
    
    def hopdict(oldhops):
        ##'relabels hoppings'
        hopdictionary=[{},{}]
        count=[0,0]
        for hop in oldhops:
            oldindex_from=hop[0]
            layer_from=hop[4][0]-1
            oldindex_to=hop[1]
            layer_to=hop[4][1]-1
            ## now check if these exist in
            fromdict=hopdictionary[layer_from]
            todict=hopdictionary[layer_to]
            if oldindex_from not in fromdict.keys():
                hopdictionary[layer_from][oldindex_from]=count[layer_from]
                count[layer_from]+=1
            if oldindex_to not in todict.keys():
                hopdictionary[layer_to][oldindex_to]=count[layer_to]
                count[layer_to]+=1
        return hopdictionary
    
    
    hops1=get_hoppings(1,1)
    hops2=get_hoppings(2,1)
    hops11=get_hoppings(1,2)
    hops22=get_hoppings(2,2)
    hops12=get_hoppings(12,1)

    
    allhops=np.concatenate((hops1,hops2,hops11,hops22,hops12))
    
    labeldict=hopdict(allhops)
    singlelayerbasissize=(v1**2+v2**2)
    dict1=labeldict[0]
    dict2=labeldict[1]
    
    for label in dict2.keys():
        dict2[label] += singlelayerbasissize
    for hop in hops1:
        hop[0],hop[1]=dict1[hop[0]],dict1[hop[1]]
        if hop[2] == None:
            if hop[3] == [1,0]: hop[2] = "+x"
            if hop[3] == [0,1]: hop[2] = "+y"
    for hop in hops2:
        hop[0],hop[1]=dict2[hop[0]],dict2[hop[1]]
        if hop[2] == None:
            if hop[3] == [1,0]: hop[2] = "+x"
            if hop[3] == [0,1]: hop[2] = "+y"
    for hop in hops12:
        hop[0],hop[1]=dict1[hop[0]],dict2[hop[1]]
        if hop[2] == None:
            if hop[3] == [1,0]: hop[2] = "+x"
            if hop[3] == [0,1]: hop[2] = "+y"
    
    ## DO THE SAME FOR NNN HOPPINGS
    
    for hop in hops11:
        hop[0],hop[1]=dict1[hop[0]],dict1[hop[1]]
    for hop in hops22:
        hop[0],hop[1]=dict2[hop[0]],dict2[hop[1]]
    
    def get_key(val,my_dict): 
        for key, value in my_dict.items(): 
             if val == value: 
                 return key 
        return "key doesn't exist"
               
    def plothopmap(hopstoplot,plottitle):
        fig=plt.figure()
        plotbase(fig)
        
        for hop in hopstoplot:
            if hop[4][0] == 1: 
                luc=largeruc1
                depdict=dict1
                textcolor="C0"
            if hop[4][0] == 2: 
                luc=largeruc2
                depdict=dict2
                textcolor="C1"
            hopindex=get_key(hop[0],depdict)
            depx,depy=luc[hopindex]
            if hop[4][1] == 1: 
                luc=largeruc1
                desdict=dict1
            if hop[4][1] == 2: 
                luc=largeruc2
                desdict=dict2
            hopindex=get_key(hop[1],desdict)
            desx,desy=luc[hopindex]+np.array(hop[3])*unitcellsize
            dx=desx-depx
            dy=desy-depy
            dep_in=inunit(depx,depy)
            des_in=inunit(desx,desy)
            normalwidth=0.03
            if hop[4][0] != hop[4][1]:
                normalwidth=0.04*hop[2]
                if des_in:
                    plt.text(desx+0.05*a,desy-0.05*a,hop[1],color="black")
            if dep_in:
                plt.text(depx+0.05*a,depy+0.05*a,hop[0],color=textcolor)
                
                if not (dx == 0.0 and dy == 0.0):
                    plt.arrow(depx,depy,dx,dy,width=normalwidth,length_includes_head=True,alpha=0.2)
        plt.xlim(-0.2*uc_size_x,1.2*uc_size_x)
        plt.ylim(-0.2*uc_size_y,1.2*uc_size_y)
        plt.title(plottitle)

    if plothoppingmaps:
        plothopmap(hops1,"NN terms, layer 1")
        plothopmap(hops2,"NN terms, layer 2")
        plothopmap(hops11,"NNN terms, layer 1")
        plothopmap(hops22,"NNN terms, layer 2")
        plothopmap(hops12,"interlayer terms, layer 1-> 2")

    return hops1,hops2,hops12,hops11,hops22


#hops1,hops2,hops12,hops11,hops22 = generate(a=1.0,d=2.22,vec=[0,1],interlayerNNnumber=10,interlayerNNdistance=7.8,offsetamount=0,plothoppingmaps=True,rho=0.39,nx=1,ny=1)




