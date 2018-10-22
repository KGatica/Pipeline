#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:59:05 2018

@author: rh
"""

## Testing clustering by density peaks on retina data
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import fcd
from sklearn.decomposition import PCA

import ens_detect_main as dclus
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

#%% This is only for parallel processing
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# num_processes = MPI.COMM_WORLD.size
# rank = MPI.COMM_WORLD.rank

#%%

Path='datasets/'
fileName="HBIhC-A15000-s5-10-g0.0177828"


Vfilt=np.load(Path+fileName+".npy")
nnodes=Vfilt.shape[1]

runInt=0.004  # Intervalo de muestreo en segundos
runTime=Vfilt.shape[0]*runInt  #Tiempo total de simulación

# Calculo de fase instantanea (ojo que la señal ya viene filtrada en la banda de ~10 Hz,
# si no, habría que filtrar primero)
phase=np.angle(signal.hilbert(Vfilt,axis=0))  

bound1=int(1/runInt)
bound2=int((runTime-1)/(runInt))
phase=phase[bound1:bound2].T  #Eliminamos el primer y ultimo segundo de la señal

runTime=Vfilt.shape[0]*runInt  #Volvemos a calcular el Tiempo total de simulación
Trun=np.arange(0,runTime,runInt)  # Vector con valores de tiempo

#%%

# Calculo de sincronía de fase total (población completa)
PLVtot=np.zeros((nnodes,nnodes))        
for ii in range(nnodes):
    for jj in range(ii):
        PLVtot[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(phase[[ii,jj],:],axis=0))))
        
phasesynch=np.abs(np.mean(np.exp(1j*phase),0))
MPsync=np.mean(phasesynch)  #Media de la fase en el tiempo
VarPsync=np.var(phasesynch)  #Varianza de la fase en el tiempo


#%% Calculo de FCs y FCD
# FCD = Matriz de FCD
# Pcorr = matriz conteniendo las FCs estiradas

# Parámetros para el cálculo de las FCs y FCD
WW = 200 #Ancho de ventana a utilizar (en muestras)
mode = 'psync'# medida para las FCs. 'psync' = sincroná de fase (en pares)
modeFCD='clarksondist' # medida para la FCD. 'clarksondist' = distancia angular


FCD,Pcorr,shift=fcd.extract_FCD(phase[:,::],maxNwindows=2000,wwidth=WW,olap=0.5,mode=mode,modeFCD=modeFCD)

#Varianza de los valores encontrados en la FCD
VarFCD = np.var(FCD[np.tril_indices(len(Pcorr),k=-5)])

# np.save(Path+fileName+"FCD%g.npy"%(WW),FCD)
# np.save(Path+fileName+"FCs%g.npy"%(WW),Pcorr)

#%% Calculo de PCA sobre la matriz de FCs
pcac = PCA()
pcac.fit(Pcorr)
U = pcac.components_.T #en caso de que quede negativo revisar y multiplicar por -1
cond1 = np.sum(pcac.explained_variance_ratio_[0:10])
allvar = pcac.explained_variance_ratio_



#%% Grafico de las 'ncomp' primeros eigenvectors (en forma de matriz)
ncomp = 6

vmax=np.max(np.abs(U[:,:ncomp]))

plt.figure(1,figsize=(10,1.5));
plt.clf()
for k in range(ncomp):
    Conect1 = np.zeros((nnodes,nnodes))
    cont = 0
    for i in range(1,nnodes):
       for j in range(0,i):
           Conect1[i,j] = U[cont,k]
           cont+=1
    ConectF = Conect1 + Conect1.T 
    axi = plt.subplot(1,ncomp+1,k+1)
    axi.set_xticklabels((),())
    axi.set_yticklabels((),())
    plt.imshow(ConectF,cmap='jet',vmin=-vmax,vmax=vmax)  
axcb=plt.subplot(1,ncomp+1,k+2)
axcb.set_aspect(6)
plt.colorbar(mappable=axi.get_images()[0],cax=axcb,panchor=(0,0.2))

            
plt.tight_layout(pad=0.2,w_pad=0.1)
# plt.savefig(Path+fileName+'pca%g.png'%WW,dpi=300)

#%% Plot de algunas FCs, de 5 en 5
plt.figure(2)
plt.clf()
IndexFc = np.arange(0,(Pcorr.shape[0]+1),5)
ColFc = 10
RowFc = int(IndexFc.shape[0]/ColFc)+1
FCMatrix = np.zeros((nnodes,nnodes,len(IndexFc)))
for k,n in enumerate(IndexFc):
    axi = plt.subplot(RowFc,ColFc,k+1)
    FCt = np.zeros((nnodes,nnodes))
    FCt[np.tril_indices(nnodes,k=-1)] = Pcorr[n]
    if mode == 'clark':
        FCreconst = FCt+FCt.T
    else:
        FCreconst = FCt+FCt.T+np.eye(nnodes)
    plt.imshow(FCreconst,cmap='jet')
    #axi.set_title('FCt=%s'%n)
    axi.set_xticklabels((),())
    axi.set_yticklabels((),())
    axi.grid('off')
    FCMatrix[:,:,k] = FCreconst
# plt.savefig(Path+fileName+'FCs%g.png'%WW,dpi=200)

#%%  Clustering de FCs
#Parametros del clustering
varexp = 0.6   #umbral de varianza explicada para seguir con clustering
npcs = 5       # numero de PCs a considerar
dc = 0.02    #   parametro dc del clustering
alpha = 0.02   #Umbral de confianza para el fit powerlaw
# Computing PCA
pca = PCA(n_components=npcs)
pca.fit(Pcorr)#pca to FCs
pcs = pca.components_

projData=pca.fit_transform(Pcorr)
# Computing distance matrix on PC space
distmat = distance.cdist(projData, projData, 'euclidean')  #Matriz de 
# distmat = FCD[indw]
rho = dclus.compute_rho(distmat,dc)
delta = dclus.compute_delta(distmat, rho)

if np.sum(FCD>0.05): 
    if cond1 >=varexp:
        # Clustering: Computing thresholds, finding centroids and assigning variables to clusters
        nclus,cluslabels,centid,threshold = dclus.find_centroids_and_cluster(distmat,rho,delta,alpha)    
        nclusters = nclus
        fig4=plt.figure(4,figsize=(10,12))
        plt.clf()
        plt.subplot(321) # delta vs rho
        plt.plot(rho,delta,'b.')
        for i in range(nclus):
            plt.plot(rho[centid==i+1],delta[centid==i+1],'o')
        plt.plot(threshold[0,:],threshold[1,:],'k.')
        plt.title(str(nclus)+' clusters')
        plt.xlabel(R'$\rho$')
        plt.ylabel(R'$\delta$')
        
        plt.subplot(323)
        #plt.plot(np.arange(len(cluslabels)),cluslabels,'.')
        for i in range(nclus):
            plt.plot(np.where(cluslabels==i+1)[0],cluslabels[cluslabels==i+1],'o')
        
        ax1=plt.subplot(322,projection='3d') # pc space
        for i in range(nclus):
            plt.plot(pcs[0,cluslabels==i+1],pcs[1,cluslabels==i+1],pcs[2,cluslabels==i+1],'.')
        plt.xlim([0.9*pcs[0,:].min(),1.1*pcs[0,:].max()])
        ax1.set_zlim([0.9*pcs[1,:].min(),1.1*pcs[1,:].max()])
        
        plt.subplot(324)
        plt.imshow(distmat,cmap='jet')
        plt.colorbar()
        
        axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
        axFC.imshow(Pcorr.T,vmin=0,vmax=1,cmap='jet',aspect='auto')
        axFC.set_xticklabels(())
        
        axClus=fig4.add_axes((0.15,0.12,0.7,0.02))
        axClus.imshow(cluslabels[None,:]*np.ones((3,1)),cmap='tab10',aspect='auto',vmin=1,vmax=10)
        axClus.set_xticklabels(())
        axClus.set_xticks(())
        
        axPCA=fig4.add_axes((0.15,0.07,0.7,0.04))
        axPCA.imshow(pcs,cmap='jet',aspect='auto')
        
                                
        plt.imshow
        
        plt.show()
        # plt.title('Hay %s ventanas consecutivas muy diferentes'%len(Higdif))
        # plt.savefig(Path+fileName+"-clust.png",dpi=200)  
        plt.close(4)
        plt.figure(5)
        plt.clf()
        for k in range(1,nclus+1):
        #    plt.figure()
            Indclus = np.where(cluslabels==k)[0]
            Indcol = 10
            Indrow = int(Indclus.shape[0]/Indcol)+1
            Matrixclus = np.zeros((nnodes,nnodes,len(Indclus)))
            for i,j in enumerate(Indclus):
                Clusi = np.zeros((nnodes,nnodes))
                Clusi[np.tril_indices(nnodes,k=-1)]=Pcorr[j]
                if mode == 'clark':
                    ClusiF = Clusi+Clusi.T
                else:    
                    ClusiF = Clusi+Clusi.T+np.eye(nnodes)
                Matrixclus[:,:,i] = ClusiF
            MedianMatrix = np.median(Matrixclus,axis=2)
            axi = plt.subplot(1,nclus,k)
            plt.imshow(MedianMatrix,cmap='jet')
            axi.set_xticklabels((),())
            axi.set_yticklabels((),())
        # plt.savefig(dir_aux+mode+'Allnodes%sMedian-%02d-'%(wwidth0,FileIND)+'g'+FileInit[9:]+'.png',dpi=200)
        np.savetxt(Path+fileName+'Cluslabels%g.txt'%WW,cluslabels,fmt='%d')
    else:
        nclusters = 0

        fig4=plt.figure(4,figsize=(6,10))
        plt.clf()
        
        ax1=plt.subplot(311,projection='3d') # pc space
        plt.plot(pcs[0,:],pcs[1,:],pcs[2,:],'.')
        plt.xlim([0.9*pcs[0,:].min(),1.1*pcs[0,:].max()])
        ax1.set_zlim([0.9*pcs[1,:].min(),1.1*pcs[1,:].max()])
        
        plt.subplot(312)
        plt.imshow(distmat,cmap='jet')
        plt.colorbar()
        
        axFC=fig4.add_axes((0.15,0.15,0.7,0.2))
        axFC.imshow(Pcorr.T,vmin=0,vmax=1,cmap='jet',aspect='auto')
        axFC.set_xticklabels(())
        
        axPCA=fig4.add_axes((0.15,0.08,0.7,0.05))
        axPCA.imshow(pcs,cmap='jet',aspect='auto')
        
        plt.show()
        # plt.savefig(Path+fileName+"-clust.png",dpi=200)  
        plt.close(4)  
else:
    nclusters = 1
# =============================================================================
#             Create .txt to quantify clusters and eigenvals to pca
# =============================================================================
# file_aux3 = mode+'%sParametersLast%02d'%(wwidth0,FileIND)
print('hay n clus',G,nclusters)
# libFCD.savetxtCluster(G,nclus,dir_aux,file_aux3)
aux_eig = [sum(allvar[0:j]) for j in range(len(allvar)+1)]
if (aux_eig[-1])>=varexp:
    n_eig[indw] = int(np.where(np.array(aux_eig)>=varexp)[0][0]+1)
else:
    n_eig[indw] = 0 #ie no hay!
# file_aux4 = mode+'eigval'+'%sParametersLast%02d'%(wwidth0,FileIND)
# libFCD.savetxtCluster(G,eig_val,dir_aux,file_aux4)
    
Tini=1
Tfin=Trun[-1]/1000 - 1
 #               CM = np.loadtxt(dir_aux + "SCmatrix.txt")

plt.figure(6,figsize=(10,14))
plt.clf()
    
plt.subplot2grid((6,5),(0,0),rowspan=2,colspan=5)
plt.plot(Trun[bound1:bound2],phasesynch)
plt.title('mean P sync')
    
plt.subplot2grid((6,5),(2,0),rowspan=2,colspan=2)
plt.imshow(FCD[1],vmin=0,vmax=0.7,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
plt.title('Sync FCD - w%g'%WW[1])
plt.grid()

plt.subplot2grid((6,5),(2,2),rowspan=2,colspan=2)
plt.imshow(FCD[0],vmin=0,vmax=0.7,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
plt.title('sync FCD - w%g'%WW[0])
plt.grid()
    
plt.subplot2grid((6,5),(4,4))
#                plt.imshow(CM,cmap='gray_r',interpolation='none')
plt.gca().set_xticklabels((),())
plt.gca().set_yticklabels((),())
plt.title('SC')
plt.grid()
    
plt.subplot2grid((6,5),(5,4))
plt.imshow(PLVtot+PLVtot.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
plt.gca().set_xticklabels((),())
plt.gca().set_yticklabels((),())
plt.title('total PLV')
plt.grid()
    
ax=plt.subplot2grid((6,5),(2,4))
ax.hist(FCD[0][np.tril_indices(len(Pcorr[0]),k=-4)],range=(0,0.25),color='C1')
ax.text(0.5,0.97,'%.4g'%VarFCD[1],transform=ax.transAxes,ha='center',va='top',fontsize='xx-small')
    
ax=plt.subplot2grid((6,5),(3,4))
ax.hist(FCD[1][np.tril_indices(len(Pcorr[1]),k=-4)],range=(0,0.25),color='C1')
ax.text(0.5,0.97,'%.4g'%VarFCD[0],transform=ax.transAxes,ha='center',va='top',fontsize='xx-small')
    
windows=[int(len(Pcorr[0])*f) for f in (0.2, 0.4, 0.6, 0.8)]
axes2=[plt.subplot2grid((6,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3))]
for axi,ind in zip(axes2,windows):
    corrMat=np.zeros((nnodes,nnodes))
    corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[0][ind]
    corrMat+=corrMat.T
    corrMat+=np.eye(nnodes)
        
    axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
    axi.set_xticklabels((),())
    axi.set_yticklabels((),())
    
    axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorr[0])))
    axi.grid()
    
windows=[int(len(Pcorr[1])*f) for f in (0.2, 0.4, 0.6, 0.8)]
axes2=[plt.subplot2grid((6,5),pos) for pos in ((5,0),(5,1),(5,2),(5,3))]
for axi,ind in zip(axes2,windows):
    corrMat=np.zeros((nnodes,nnodes))
    corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[1][ind]
    corrMat+=corrMat.T
    corrMat+=np.eye(nnodes)
        
    axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
    axi.set_xticklabels((),())
    axi.set_yticklabels((),())
    axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorr[1])))
    axi.grid()

plt.tight_layout()
    
plt.savefig(dir_aux+"sync%02d-g%g.png"%(FileIND,float(G)),dpi=200)
    

    
# =============================================================================
#             Create .txt
# =============================================================================
file_global = 'GlobalParameters%02d'%FileIND
with open(dir_aux+file_global+".txt",'w') as dataf:
    dataf.write('G PsynchMean PsyncVar PLVMean PLVVar VarFCD VarFCD2'+ "\n")
    dataf.write("%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n"%(float(G),MPsync,VarPsync,MPsync,VarPsync,
                                                VarFCD[0],VarFCD[1],nclusters[0],nclusters[1],
                                                n_eig[0],n_eig[1]))



