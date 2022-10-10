#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:04:17 2021

@author: parthahazarika

This python code normalizes the H&E stained images.

Workflow based on the following papers:
A method for normalizing histology slides for quantitative analysis. 
M. Macenko et al., ISBI 2009
    http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

#INPUT RGB IMAGE 

import os

directory = '/Users/parthahazarika/Desktop/MoNuSegTrainingData-HnE-Normalization/Tissue Images'
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        newfilename = os.path.join(directory, filename)
        print(newfilename)
        img=cv2.imread(newfilename, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        Io = 240 
        alpha = 1  
        beta = 0.15 


        #Step 1: Convert RGB to OD 
        
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])
        
        h, w, c = img.shape
        
        img = img.reshape((-1,3))
        
        OD = -np.log10((img.astype(np.float)+1)/Io) #for opencv imread (log 0 is indeterminate if any pixel has value 0 hence add 1 )
    
        #Step 2: Remove data with OD intensity less than β 
       
        ODhat = OD[~np.any(OD < beta, axis=1)]
       
        
        #Step 3: Calculate SVD on the OD tuples 
        
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        
        #Step 4: Create plane from the SVD directions with two largest values 
          
        That = ODhat.dot(eigvecs[:,1:3]) #Dot product
        
        #Step 5: Project data on the plane & normalize 
        #Step 6: Calculating angle of each point 
        #Find out the min and max vectors and project back to OD space
        phi = np.arctan2(That[:,1],That[:,0])
        
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)
        
        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        
      
        if vMin[0] > vMax[0]:    
            HE = np.array((vMin[:,0], vMax[:,0])).T
            
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T
        
        
        #rows -(RGB), columns- OD values
        Y = np.reshape(OD, (-1, 3)).T
        
        #concentrations of the individual stains
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]
        
        #normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])
        
        #Step 8: Convert extreme values back to OD space
        
        
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
        
        # Separating H and E components
        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
        
        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
        
        hnfilename = newfilename.replace('.tif', '-HnE-Normalized.tiff')
        hnnewfilename = os.path.join(directory, hnfilename)
        plt.imsave(hnnewfilename, Inorm)
       