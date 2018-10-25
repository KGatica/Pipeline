# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:57:38 2016

@author: jmaidana
@author: porio
"""
from __future__ import division
import numpy as np
#import seaborn as sns ## is used to plot some of the results
#import pandas as pd ## pandas has a function to make the correlation between multiple time series
import matplotlib.pylab as plt

from numpy import linalg as LA

def ccorrcoef(alpha1, alpha2, axis=None):
    if axis is not None and alpha1.shape[axis] != alpha2.shape[axis]:
        raise(ValueError, "shape mismatch")
    # compute mean directions
    if axis is None:
        n = alpha1.size #toma el largo de la matriz
    else:
        n = alpha1.shape[axis]#En caso de operar sobre fila o columna
    #################################################################
    c1 = np.cos(alpha1)
    c1_2 = np.cos(2*alpha1)
    c2 = np.cos(alpha2)
    c2_2 = np.cos(2*alpha2)
    s1 = np.sin(alpha1)
    s1_2 = np.sin(2*alpha1)
    s2 = np.sin(alpha2)
    s2_2 = np.sin(2*alpha2)

    sumfunc = lambda x: np.sum(x, axis=axis)
    num = 4 * (sumfunc(c1*c2) * sumfunc(s1*s2) -
               sumfunc(c1*s2) * sumfunc(s1*c2))
    
    den = np.sqrt((n**2 - sumfunc(c1_2)**2 - sumfunc(s1_2)**2) *
                  (n**2 - sumfunc(c2_2)**2 - sumfunc(s2_2)**2))

    rho = num / den

    return rho


def phaseScramble(data,Nsurr=10):
    len_d=len(data)
    fftdata=np.fft.fft(data)
    angles=np.angle(fftdata)
    amplitudes=np.abs(fftdata)
    
    surrAngles=np.random.uniform(low=-np.pi,high=np.pi,size=(Nsurr,len(angles)))
    surrAngles[:,1:len_d//2]=surrAngles[:,-1:len_d//2:-1]
    surrAngles[:,len_d//2]=0
    
    fftSurr=amplitudes*(np.cos(surrAngles) + 1j*np.sin(surrAngles))
    surrData=np.fft.ifft(fftSurr,axis=-1)
    
    return surrData


def extract_FCD(data,wwidth=1000,maxNwindows=100,olap=0.9,coldata=False,mode='corr',modeFCD='corr'):
    if olap>=1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data=data.T    
    
    all_corr_matrix = []
    lenseries=len(data[0])
    Nwindows=min(((lenseries-wwidth*olap)//(wwidth*(1-olap)),maxNwindows))
    shift=int((lenseries-wwidth)//(Nwindows-1))
    if Nwindows==maxNwindows:
        wwidth=int(shift//(1-olap))
    
    indx_start = range(0,(lenseries-wwidth+1),shift)
    indx_stop = range(wwidth,(1+lenseries),shift)
         
    nnodes=len(data)
    #    mat_ones=np.tril(-10*np.ones((n_ts,n_ts))) #this is a condition to then eliminate the diagonal
    for j1,j2 in zip(indx_start,indx_stop):
        aux_s = data[:,j1:j2]
        if mode=='corr':
            corr_mat = np.corrcoef(aux_s) 
        elif mode=='psync':
            corr_mat=np.mean(np.abs((np.exp(1j*aux_s[:,None,:])+np.exp(1j*aux_s[None,:,:]))/2),-1)
            # for ii in range(nnodes):
            #     for jj in range(ii):
            #         corr_mat[ii,jj]=np.mean(np.abs(np.mean(np.exp(1j*aux_s[[ii,jj],:]),0)))
        elif mode=='pcoher': #PLV phase locking value
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(aux_s[[ii,jj],:],axis=0))))
        elif mode=='tdcorr': #time-delayed correlation
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    maxCorr=np.max(np.correlate(aux_s[ii,:],aux_s[jj,:],mode='full')[wwidth//2:wwidth+wwidth//2])
                    corr_mat[ii,jj]=maxCorr/np.sqrt(np.dot(aux_s[ii,:],aux_s[ii,:])*np.dot(aux_s[jj,:],aux_s[jj,:]))
        elif mode == 'circcorr':
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):   
                    corr_mat[ii,jj] = ccorrcoef(aux_s[ii,:],aux_s[jj,:])
        elif mode == 'clark':#clarkson para las FC
            corr_mat=np.zeros((nnodes,nnodes))
            for ii in range(nnodes):
                for jj in range(ii):
                    corr_mat[ii,jj]=LA.norm(aux_s[ii,:]/LA.norm(aux_s[ii,:]) - aux_s[jj,:]/LA.norm(aux_s[jj,:]))  

        all_corr_matrix.append(corr_mat)
        
    corr_vectors=np.array([allPm[np.tril_indices(nnodes,k=-1)] for allPm in all_corr_matrix])
    L = np.shape(corr_vectors)[0]
    FCD = np.zeros((L,L))
    if modeFCD == 'corr':
        CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
        FCD = 1 - np.abs(np.corrcoef(CV_centered))
    elif modeFCD == 'angdist':#angular distance
        for ii in range(L):
            for jj in range(ii):
                FCD[ii,jj]=np.arccos((np.dot(corr_vectors[ii,:],corr_vectors[jj,:]))/(LA.norm(corr_vectors[ii,:])*LA.norm(corr_vectors[jj,:])))/np.pi  
                FCD[jj,ii]=FCD[ii,jj]
    elif modeFCD == 'clarksondist':#angular distance by clarkson
        for ii in range(L):
            for jj in range(ii):
                FCD[ii,jj]= LA.norm(corr_vectors[ii,:]/LA.norm(corr_vectors[ii,:]) - corr_vectors[jj,:]/LA.norm(corr_vectors[jj,:]))  
                FCD[jj,ii]=FCD[ii,jj]
           
    
    #return CV_centered,corr_vectors,shift
    return FCD,corr_vectors,shift




#%%
## As an example it takes the next time series:

if __name__=='__main__':    
    import Wavelets      
#    data_series=np.loadtxt("Vfilt-FR30to45noIh-50nodes-seed619-g0.316228.txt.gz")
    data_series=np.loadtxt("Vfilt-FR30to45chaos2C-50nodes-seed213-g0.01.txt.gz")

#    data_series=data_series[::10,:]
    dt=0.004
    runTime=27
    nnodes=50
    Trun=np.arange(0,runTime,dt)
    
    freqs=np.arange(2,15,0.2)  #Desired frequencies
    Periods=1/(freqs*dt)    #Desired periods in sample untis
    dScales=Periods/Wavelets.Morlet.fourierwl  #desired Scales
    
    #wavel=Wavelets.Morlet(EEG,largestscale=10,notes=20,scaling='log')
    wavelT=[Wavelets.Morlet(y1,scales=dScales) for y1 in data_series.T]
    cwt=np.array([wavel.getdata() for wavel in wavelT])
    pwr=np.array([wavel.getnormpower() for wavel in wavelT])
    
    phase=np.array([np.angle(cwt_i) for cwt_i in cwt])
    
    spec=np.sum(pwr,-1)
    maxFind=np.argmax(spec,-1)
    maxFreq=freqs[maxFind]
    
    bound1=int(1/dt)
    bound2=int((runTime-1)/dt)
    phaseMaxF=phase[range(nnodes),maxFind,bound1:bound2]
    phasesynch=np.abs(np.mean(np.exp(1j*phaseMaxF),0))

    Pcoher=np.zeros((nnodes,nnodes))
    
    for ii in range(nnodes):
        for jj in range(ii):
            Pcoher[ii,jj]=np.abs(np.mean(np.exp(1j*np.diff(phaseMaxF[[ii,jj],:],axis=0))))

    #sns.clustermap(corr_data_series) #to see how it cluster the time series, seaborn has the function clustermap
    ##############################################################################
    #%%

    PcorrFCD,Pcorr,shift=extract_FCD(phaseMaxF[:,::],wwidth=100,olap=0.5,mode='psync')
    Tini=1
    Tfin=Trun[-1]/1000 - 1
    plt.figure(4,figsize=(10,12))
    plt.clf()
    
    plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=5)
    plt.plot(Trun[bound1:bound2],phasesynch)
    plt.title('mean P sync')
    
    plt.subplot2grid((5,5),(2,0),rowspan=2,colspan=2)
    plt.imshow(PcorrFCD,vmin=0,vmax=1,extent=(Tini,Tfin,Tfin,Tini),interpolation='none',cmap='jet')
    plt.title('P coher FCD')
    plt.grid()

#    plt.subplot2grid((5,5),(3,4))
#    plt.imshow(Psynch+Psynch.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
#    plt.gca().set_xticklabels((),())
#    plt.gca().set_yticklabels((),())
#    plt.title('P sync')
#    plt.grid()
    
    plt.subplot2grid((5,5),(3,4))
    plt.imshow(Pcoher+Pcoher.T+np.eye(nnodes),cmap='jet',vmax=1,vmin=0,interpolation='none')
    plt.gca().set_xticklabels((),())
    plt.gca().set_yticklabels((),())
    plt.title('P coher')
    plt.grid()
    
    axes2=[plt.subplot2grid((5,5),pos) for pos in ((4,0),(4,1),(4,2),(4,3),(4,4))]
    for axi,ind in zip(axes2,(20,35,50,75,90)):
        corrMat=np.zeros((nnodes,nnodes))
        corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
        corrMat+=corrMat.T
        corrMat+=np.eye(nnodes)
        
        axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
        
        axi.set_xticklabels((),())
        axi.set_yticklabels((),())
        
        axi.set_title('t=%.2g'%(ind*Tfin/len(Pcorr)))
        axi.grid()


    #correlations,corrV,delta = extract_FCD(np.unwrap(data_series,axis=0),coldata=True,maxNwindows=100,wwidth=1000,mode='corr')
    #correlations,corrV = extract_FCD(data_series,coldata=True,maxNwindows=150,wwidth=200,mode='corr')
    
