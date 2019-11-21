# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:07:44 2019

@author: Rehan
"""
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy
from mpl_toolkits.mplot3d import Axes3D

def AudioResegmentationGMM(fVects, idx, outputdir, AudioDataSet):
    # =============================================================================
    # Read the newshots pickle and results for all the speakers
    # =============================================================================
    allSpeakerShots = []
    allSpeakerMetric = []
    for f in os.listdir(outputdir):
        if f[-6::] == 'pickle':
            with open(os.path.join(outputdir,f), 'rb') as pf:
                allSpeakerShots.append(pickle.load(pf))
                print(f)
        elif f[-3::] == 'txt':
            allSpeakerMetric.append(pd.read_table(os.path.join(outputdir,f)))
            print(f)
    # =============================================================================
    # Based on high confidence values separate the speech freames (MFCC) of each user.  
    # =============================================================================
    datafortraining = [np.array([]),np.array([]),np.array([]),np.array([])]
    for j in range(4):
        ff = 0
        for i,sho in enumerate(allSpeakerShots[j]):
            if allSpeakerMetric[j].iloc[i,1] >=0 and allSpeakerMetric[j].iloc[i,1] <=3:
                if allSpeakerMetric[j].iloc[i,2] >= 1.5:
                    start = sho[0]
                    end = sho[-1]
                    if ff==0:
                        datafortraining[j] = copy.copy(fVects[:,int(start*4):int((end+1)*4)])
                    else:
                        datafortraining[j] = np.append(datafortraining[j],fVects[:,int(start*4):int((end+1)*4)], axis=1)
                    ff=1
    # =============================================================================
    # Create 4 GMM models and train on each high confidence speech frames.
    # =============================================================================
    gmm = [GaussianMixture(n_components=20, covariance_type='diag', max_iter=100) for i in range(4)]
    for i in range(4):
        gmm[i].fit(datafortraining[i].T)
        print('Data %d with %d samples' %(i,datafortraining[i].shape[1]))
    # =============================================================================
    # Find likelihood of all speech only samples.
    # =============================================================================
    fVects_speechonly = copy.copy(fVects[:,idx])
    likelihood = gmm[0].score_samples(fVects_speechonly.T)
    for g in gmm[1:]:
        likelihood = np.column_stack((likelihood, g.score_samples(fVects_speechonly.T)))
    most_likely = likelihood.argmax(axis=1)
    
    return most_likely, likelihood

def PlotTSNE(fVects, idx, outputdir, AudioDataSet):
    # =============================================================================
    # Read the newshots pickle and results for all the speakers
    # =============================================================================
    allSpeakerShots = []
    allSpeakerMetric = []
    for f in os.listdir(outputdir):
        if f[-6::] == 'pickle':
            with open(os.path.join(outputdir,f), 'rb') as pf:
                allSpeakerShots.append(pickle.load(pf))
                print(f)
        elif f[-3::] == 'txt':
            allSpeakerMetric.append(pd.read_table(os.path.join(outputdir,f)))
            print(f)
    # =============================================================================
    # Based on high confidence values separate the speech freames (MFCC) of each user.  
    # =============================================================================
    datafortraining = [np.array([]),np.array([]),np.array([]),np.array([])]
    for j in range(4):
        ff = 0
        for i,sho in enumerate(allSpeakerShots[j]):
            if allSpeakerMetric[j].iloc[i,1] >=0 and allSpeakerMetric[j].iloc[i,1] <=3:
                if allSpeakerMetric[j].iloc[i,2] >= 1.5:
                    start = sho[0]
                    end = sho[-1]
                    if ff==0:
                        datafortraining[j] = copy.copy(fVects[:,int(start*4):int((end+1)*4)])
                    else:
                        datafortraining[j] = np.append(datafortraining[j],fVects[:,int(start*4):int((end+1)*4)], axis=1)
                    ff=1
    # =============================================================================
    # Create 4 GMM models and train on each high confidence speech frames.
    # =============================================================================
    X = np.concatenate((datafortraining[0].T, datafortraining[1].T, 
                        datafortraining[2].T, datafortraining[3].T))
    Y = np.concatenate((np.zeros(datafortraining[0].shape[1]),np.ones(datafortraining[1].shape[1]),
                        np.ones(datafortraining[2].shape[1])*2,np.ones(datafortraining[3].shape[1])*3))

    print('Data %d with %d samples' %(0,datafortraining[0].shape[1]))
    print('Data %d with %d samples' %(1,datafortraining[1].shape[1]))
    print('Data %d with %d samples' %(2,datafortraining[2].shape[1]))
    print('Data %d with %d samples' %(3,datafortraining[3].shape[1]))
    # =============================================================================
    # PCA plot    
    # =============================================================================
#    pca = PCA(n_components = 3)
#    Xpca = pca.fit_transform(X)
#    print(pca.explained_variance_ratio_)
##    import seaborn as sns
##    plt.figure(figsize=(16,10))
##    sns.scatterplot(x=Xpca[:,0], y=Xpca[:,1], hue=Y, palette=sns.color_palette("hls", 4),legend="full", alpha=0.3)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(xs=Xpca[:,0], ys=Xpca[:,1], zs=Xpca[:,2], zdir='y', s=20, 
#                   c=Y, depthshade=True)
#    ax.set_xlabel('pca-one')
#    ax.set_ylabel('pca-two')
#    ax.set_zlabel('pca-three')
#    plt.show()    
    # =============================================================================
    # TSNE plot
    # =============================================================================
    tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 300)
    Xtsne = tsne.fit_transform(X)
    import seaborn as sns
    plt.figure(figsize=(16,10))
    sns.scatterplot(x=Xtsne[:,0], y=Xtsne[:,1], hue=Y, palette=sns.color_palette("hls", 4),legend="full", alpha=0.3)

    return 0

def AudioResegmentationKMeans(fVects, idx, outputdir, AudioDataSet):
    # =============================================================================
    # Read the newshots pickle and results for all the speakers
    # =============================================================================
    allSpeakerShots = []
    allSpeakerMetric = []
    for f in os.listdir(outputdir):
        if f[-6::] == 'pickle':
            with open(os.path.join(outputdir,f), 'rb') as pf:
                allSpeakerShots.append(pickle.load(pf))
                print(f)
        elif f[-3::] == 'txt':
            allSpeakerMetric.append(pd.read_table(os.path.join(outputdir,f)))
            print(f)
    # =============================================================================
    # Based on high confidence values separate the speech freames (MFCC) of each user.  
    # =============================================================================
    datafortraining = [np.array([]),np.array([]),np.array([]),np.array([])]
    for j in range(4):
        ff = 0
        for i,sho in enumerate(allSpeakerShots[j]):
            if allSpeakerMetric[j].iloc[i,1] >=0 and allSpeakerMetric[j].iloc[i,1] <=3:
                if allSpeakerMetric[j].iloc[i,2] >= 1.5:
                    start = sho[0]
                    end = sho[-1]
                    if ff==0:
                        datafortraining[j] = copy.copy(fVects[:,int(start*4):int((end+1)*4)])
                    else:
                        datafortraining[j] = np.append(datafortraining[j],fVects[:,int(start*4):int((end+1)*4)], axis=1)
                    ff=1
    # =============================================================================
    # Create KMeans models and train on each high confidence speech frames.
    # =============================================================================
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(datafortraining[0].T)

    kmeans.cluster_centers_[0] = np.mean(datafortraining[0].T, axis=0)
    print('Data %d with %d samples' %(0,datafortraining[0].shape[1]))
    kmeans.cluster_centers_[1] = np.mean(datafortraining[1].T, axis=0)
    print('Data %d with %d samples' %(1,datafortraining[1].shape[1]))
    kmeans.cluster_centers_[2] = np.mean(datafortraining[2].T, axis=0)
    print('Data %d with %d samples' %(2,datafortraining[2].shape[1]))
    kmeans.cluster_centers_[3] = np.mean(datafortraining[3].T, axis=0)
    print('Data %d with %d samples' %(3,datafortraining[3].shape[1]))
    # =============================================================================
    # Find likelihood of all speech only samples.
    # =============================================================================
    fVects_speechonly = copy.copy(fVects[:,idx])
    most_likely = kmeans.predict(fVects_speechonly.T)
    
    return most_likely

def AudioResegmentationSVM(fVects, idx, outputdir, AudioDataSet):
    # =============================================================================
    # Read the newshots pickle and results for all the speakers
    # =============================================================================
    allSpeakerShots = []
    allSpeakerMetric = []
    for f in os.listdir(outputdir):
        if f[-6::] == 'pickle':
            with open(os.path.join(outputdir,f), 'rb') as pf:
                allSpeakerShots.append(pickle.load(pf))
                print(f)
        elif f[-3::] == 'txt':
            allSpeakerMetric.append(pd.read_table(os.path.join(outputdir,f)))
            print(f)
    # =============================================================================
    # Based on high confidence values separate the speech freames (MFCC) of each user.  
    # =============================================================================
    datafortraining = [np.array([]),np.array([]),np.array([]),np.array([])]
    for j in range(4):
        ff = 0
        for i,sho in enumerate(allSpeakerShots[j]):
            if allSpeakerMetric[j].iloc[i,1] >=0 and allSpeakerMetric[j].iloc[i,1] <=3:
                if allSpeakerMetric[j].iloc[i,2] >= 1.5:
                    start = sho[0]
                    end = sho[-1]
                    if ff==0:
                        datafortraining[j] = copy.copy(fVects[:,int(start*4):int((end+1)*4)])
                    else:
                        datafortraining[j] = np.append(datafortraining[j],fVects[:,int(start*4):int((end+1)*4)], axis=1)
                    ff=1
    # =============================================================================
    # Create KMeans models and train on each high confidence speech frames.
    # =============================================================================
    X = np.concatenate((datafortraining[0].T, datafortraining[1].T, 
                        datafortraining[2].T, datafortraining[3].T))
    Y = np.concatenate((np.zeros(datafortraining[0].shape[1]),np.ones(datafortraining[1].shape[1]),
                        np.ones(datafortraining[2].shape[1])*2,np.ones(datafortraining[3].shape[1])*3))

    lin_clf = svm.LinearSVC(class_weight='balanced')
    lin_clf.fit(X, Y)

    # =============================================================================
    # Find likelihood of all speech only samples.
    # =============================================================================
    fVects_speechonly = copy.copy(fVects[:,idx])
    most_likely = lin_clf.predict(fVects_speechonly.T)
    
    return most_likely

