#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:51:45 2018

@author: kanela
Functions to create figures and text file.

graph1(runTime,Trun,bound,phasesynch,FCDi,nnodes,Psynch,PLV,Pcorri,VarFCDi,directory,file,W)
    -Inputs: runTime,Trun,bound,phasesynch,FCDi,nnodes,Psynch,PLV,Pcorri,VarFCDi,directory,file,W
     W is width of slidding windows.
     nnodes is the numbers of nodes to the network
     FCDi is an array 2D, with FCD PLV and FCD psync matrix.
    -Description:This function create a .png in directory, who shows the FCD matrix with PLV metric and FCD matrix with psync metric
     with theirs respective histrograms and some FC (PLV) and FC(psync).
    -Output: file.png

graph2(FCDi,FCsN,nnodes,directory,file,runTime,wwidth0)
    -Inputs: FCDi,FCsN,nnodes,directory,file,runTime,wwidth0
     FCDi is an array 2D, with FCD PLV and FCD psync matrix.
     wwidth is the width to slidding windows.
     FCsN is an array 2D, with FC PLV and FC psync.
     runTime is the total time simulation to create a network with number of nodes nnodes
    -Description:Shows the FCDs and clustering to FCDs (with plv and psync matrix).
     Also this function shows the rho/delta plot distances.
    -Output:file.png

savetxt(G,PsynchMean,PsyncVar,PLVMean,PLVVar,VarFCD,VarFCD2,directory,file,n)
    -Inputs: G,PsynchMean,PsyncVar,PLVMean,PLVVar,VarFCD,VarFCD2,directory,file,n
    -Description: For each iteration "n", save in "directory" a "file.txt" with G,PsynchMean,PsyncVar,PLVMean,PLVVar,VarFCD,VarFCD2
     information.
    -Output: "file.txt"
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D

def graph1(runTime,Trun,bound,phasesynch,FCDi,nnodes,Psynch,PLV,Pcorri,VarFCDi,directory,file,W):
    #Tini=0
    Tfin=runTime - 2
    plt.figure(44,figsize=(10,14))
    plt.clf()
    
    plt.subplot2grid((6,5),(0,0),rowspan=2,colspan=5)
    plt.plot(Trun[bound[0]:bound[1]],phasesynch)
    plt.title('mean P sync')

    
    plt.subplot2grid((6,5),(2,0),rowspan=2,colspan=2)
    plt.imshow(FCDi[0],vmin=0,vmax=0.25,interpolation='none',cmap='jet')#plv
    
    
    plt.title('FCD - w%s - sync'%W)
    plt.grid()

    plt.subplot2grid((6,5),(2,2),rowspan=2,colspan=2)
    plt.imshow(FCDi[1],vmin=0,vmax=0.15,interpolation='none',cmap='jet')#psync
    
    plt.title('FCD - w%s - sync'%W)
    plt.grid()
    
    
    plt.subplot2grid((6,5),(5,4))
    plt.imshow(Psynch+Psynch.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
    plt.gca().set_xticklabels((),())
    plt.gca().set_yticklabels((),())
    plt.title('P sync')
    plt.grid()
    
    ax=plt.subplot2grid((6,5),(2,4))
    plt.hist(FCDi[0][np.tril_indices(len(Pcorri[1]),k=-5)],bins=20,range=(0,0.25))#plv
    ax.text(0.5,0.97,'%.4g'%VarFCDi[0],transform=ax.transAxes,ha='center',va='top',fontsize='xx-small')
    
    ax=plt.subplot2grid((6,5),(3,4))
    plt.hist(FCDi[1][np.tril_indices(len(Pcorri[1]),k=-5)],bins=20,range=(0,0.15))#psync
    ax.text(0.5,0.97,'%.4g'%VarFCDi[1],transform=ax.transAxes,ha='center',va='top',fontsize='xx-small')
    
    plt.subplot2grid((6,5),(4,4))
    plt.imshow(PLV+PLV.T+np.eye(nnodes),cmap='jet',interpolation='none',vmax=1,vmin=0)
    plt.gca().set_xticklabels((),())
    plt.gca().set_yticklabels((),())
    plt.title('sync')
    plt.grid()
    
    windows=[int(len(Pcorri[0])*f) for f in (0.2, 0.4, 0.6, 0.8)]
    axes2=[plt.subplot2grid((6,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3))]
    for axi,ind in zip(axes2,windows):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=Pcorri[0][ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
        
        axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
#        axi.set_xticklabels((),())
#        axi.set_yticklabels((),())
#        
#        axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorri[0])))
#        axi.grid()
    
    windows=[int(len(Pcorri[1])*f) for f in (0.2, 0.4, 0.6, 0.8)]
    axes2=[plt.subplot2grid((6,5),pos) for pos in ((5,0),(5,1),(5,2),(5,3))]
    for axi,ind in zip(axes2,windows):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=Pcorri[1][ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
        
        axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
        axi.set_xticklabels((),())
        axi.set_yticklabels((),())
        axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorri[1])))
        axi.grid()

    plt.tight_layout()
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+file)
    print('sali')
    plt.close()
    return

def graph2(FCDi,FCsN,nnodes,directory,file,runTime,wwidth0):
    for indFCDi,jindFCDi in enumerate(FCDi):
        FCs = FCsN[indFCDi]
        distances = jindFCDi.copy()
        perc=20        
        print('distances',np.shape(distances))
        dc=np.percentile(distances[np.tril_indices(len(distances),k=-1)],perc)    
        rho=np.zeros(len(FCs))
        for i in range(len(FCs)):
            rho[i]=np.sum(distances[i,:]<dc)        
        delta=np.zeros(len(FCs))
        for i in range(len(FCs)):
            if rho[i]==np.max(rho):
                delta[i]=np.max(distances[i,:])
            else:
                delta[i]=np.min(distances[i,rho>rho[i]])    
                
        if np.array_equal(rho,len(FCs)*np.ones(len(FCs))):
            centroids=np.array([np.argmax(delta)])    
        else:
            centroids=np.where((delta>(max(delta)*0.75))*(rho>(max(rho)*0.75)))[0]#buscar cuando se cumplen estas condiciones
            
        cluster=-1*np.ones(len(FCs),dtype=int)        
        for i in range(len(centroids)):
            cluster[centroids[i]]=int(i)    
        ordrho=np.argsort(-rho)
        
        for i in ordrho:
            if cluster[i]==-1:
                if len(distances[i,rho>rho[i]])==0:
                    cluster[i]=cluster[np.where(distances[i,:]==np.min(distances[i,rho==rho[i]]))]
                    if cluster[i]==-1:
                        cluster[i]=cluster[np.where(distances[i,:]==np.min(distances[i,centroids]))]
                else:
                    cluster[i]=cluster[np.where(distances[i,:]==np.min(distances[i,rho>rho[i]]))]
        
        outliermultiplierdelta=0.3
        outliermultiplierrho=0.3
        
        outliers=[]
        for i in range(len(FCs)):
            if delta[i]>outliermultiplierdelta*max(delta) and rho[i]<outliermultiplierrho*max(rho):
                outliers.append(i)
                cluster[i]=-1
        
        if len(centroids)==1:
            if len(outliers)==1:
                print('Hay ' + str(len(centroids)) + ' cluster y ' + str(len(outliers)) + ' outlier')
            else:
                print('Hay ' + str(len(centroids)) + ' cluster y ' + str(len(outliers)) + ' outliers')
        else:
            if len(outliers)==1:
                print('Hay ' + str(len(centroids)) + ' clusters y ' + str(len(outliers)) + ' outlier')
            else:
                print('Hay ' + str(len(centroids)) + ' clusters y ' + str(len(outliers)) + ' outliers')
        
        pca=PCA(n_components=3)
        pca.fit(np.array(FCs))
        transfX=pca.transform(FCs)
        

#        if indFCDi == 0:
        plt.figure(figsize=(12,9))
        plt.clf()        
        ax = plt.subplot2grid((3,4),(0,2*indFCDi+1),rowspan=3)
        print('uno', 0,2*indFCDi+1)
        for c in range(len(centroids)):
    
            plt.plot(rho[cluster==c],delta[cluster==c],'.')
        plt.plot(rho[outliers],delta[outliers],'.')
        plt.xlabel(R'$\rho$')
        plt.ylabel(R'$\delta$')
        ax.axes.get_yaxis().set_visible(False)
        
        #ax.zaxis.set_ticks_position('right') 
        plt.title(R'$\rho$ $\delta$ FCD W%s psync'%wwidth0 )
        
        plt.subplot2grid((3,4),(0,2*indFCDi))
        
        if indFCDi == 0:
            plt.imshow(FCDi[indFCDi],vmin=0,vmax=0.25,interpolation='none',cmap='jet')
            plt.title('FCD W%s clarkson'%wwidth0)
        else:
            plt.imshow(FCDi[indFCDi],vmin=0,vmax=0.15,interpolation='none',cmap='jet')
            plt.title('FCD W%s clarkson'%wwidth0)
        plt.subplot2grid((3,4),(1,2*indFCDi),projection='3d')
        plt.axis('equal')
        
        plt.scatter(x=transfX[:,0],y=transfX[:,1],zs=transfX[:,2],s=10,c=np.linspace(0,1,num=len(transfX)))
        plt.plot(transfX[:,0],transfX[:,1],transfX[:,2],'-',lw=0.5)
        
        plt.subplot2grid((3,4),(2,2*indFCDi),rowspan=2,projection='3d')
        
        for i in range(len(centroids)):
            plt.scatter(x=transfX[cluster==i,0],y=transfX[cluster==i,1],zs=transfX[cluster==i,2],s=10,c=np.linspace(0,1,num=len(transfX[cluster==i,0])))
        plt.scatter(x=transfX[outliers,0],y=transfX[outliers,1],zs=transfX[outliers,2],s=10,c=np.linspace(0,1,num=len(transfX[outliers,0])))
#        else:
#            axx = plt.subplot2grid((3,4),(0,2),rowspan=3)
#            for c in range(len(centroids)):
#        
#                plt.plot(rho[cluster==c],delta[cluster==c],'.')
#            plt.plot(rho[outliers],delta[outliers],'.')
#            plt.xlabel(R'$\rho$')
#            plt.ylabel(R'$\delta$')
#            axx.axes.get_yaxis().set_visible(False)
#            
#            #ax.zaxis.set_ticks_position('right') 
#            #plt.title(R'$\rho$ $\delta$' )
#            plt.title(R'$\rho$ $\delta$ FCD W%s psync'%wwidth0 )
#            
#            plt.subplot2grid((3,4),(0,3))
#            if indFCDi == 0:
#                plt.imshow(FCDi[indFCDi],vmin=0,vmax=0.25,interpolation='none',cmap='jet')
#                plt.title('FCD W%s corr'%wwidth0)
#            else:
#                plt.imshow(FCDi[indFCDi],vmin=0,vmax=0.15,interpolation='none',cmap='jet')
#                plt.title('FCD W%s clarkson'%wwidth0)
#            plt.subplot2grid((3,4),(1,3),projection='3d')
#            plt.axis('equal')
#            
#            plt.scatter(x=transfX[:,0],y=transfX[:,1],zs=transfX[:,2],s=10,c=np.linspace(0,1,num=len(transfX)))
#            plt.plot(transfX[:,0],transfX[:,1],transfX[:,2],'-',lw=0.5)
#            
#            plt.subplot2grid((3,4),(2,3),rowspan=2,projection='3d')
#            
#            for i in range(len(centroids)):
#                plt.scatter(x=transfX[cluster==i,0],y=transfX[cluster==i,1],zs=transfX[cluster==i,2],s=10,c=np.linspace(0,1,num=len(transfX[cluster==i,0])))
#            plt.scatter(x=transfX[outliers,0],y=transfX[outliers,1],zs=transfX[outliers,2],s=10,c=np.linspace(0,1,num=len(transfX[outliers,0])))

    plt.savefig(directory+file)  
    plt.close()
    return
        
def savetxt(G,PsynchMean,PsyncVar,PLVMean,PLVVar,VarFCD,VarFCD2,directory,file):
    with open(directory+file+".txt",'w') as dataf:
        dataf.write('G PsynchMean PsyncVar PLVMean PLVVar VarFCD VarFCD2'+ "\n")
        print(type(G),type(PsynchMean),type(PsyncVar),type(PLVMean),type(PLVVar),type(VarFCD),type(VarFCD2))
        dataf.write("%g\t%g\t%g\t%g\t%g\t%g\t%g\n"%(float(G),PsynchMean,PsyncVar,PLVMean,PLVVar,VarFCD,VarFCD2))

def savetxtCluster(G,nclus,directory,file):
    with open(directory+file+".txt",'w') as dataf:
        dataf.write('G numClus'+ "\n")
        dataf.write("%g\t%g\n"%(float(G),nclus))   