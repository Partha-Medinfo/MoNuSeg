#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:24:37 2021

@author: parthahazarika

The program will do the watershedding of the input image (output from the model) for instance segmentation

"""

import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from skimage import measure, color, io

img = cv2.imread('/Users/parthahazarika/Desktop/Test Data/Image/output_mask.jpg')  # Read as 3 channels
img_grey = img[:,:,0]
#plt.imshow(img_grey)
#plt.imshow(img_grey, cmap = 'gray')

#Threshold image to binary using OTSU. 
r1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap = 'gray')

#docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
#Opening - erosion followed by dilation
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#plt.imshow(opening, cmap = 'gray')

#Identifying sure background area, dilating pixels of the foreground object and remaining is the background
#The area between sure background and foreground is the uknown area. 
#This area will find by the watershed 
sure_bg = cv2.dilate(opening,kernel,iterations=10)
#plt.imshow(sure_bg, cmap = 'gray')

#Finding the sure foreground area using distance transform 
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

#LThreshold the distance by 20%
r2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
#plt.imshow(sure_fg, cmap = 'gray')

#Unknown region is  = (bkground - foreground)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#plt.imshow(unknown, cmap = 'gray')

#Creating a marker and label the regions inside. i.e., sure foreground 
ret3, markers = cv2.connectedComponents(sure_fg)
#plt.imshow(markers, cmap = 'gray')

#Now the entire background pixels is assigned to 0.
#So, the watershed will consider this region as unknown.
#Hence adding 10 to all labels so that sure background values become 10 instead of 0
markers = markers+10

# Now, change the value of pixels of the unknown region to 0
markers[unknown==255] = 0
#plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

#Watershed filling. Return a value of -1, i.e., the boundary region will be marked as -1
markers = cv2.watershed(img, markers)
#plt.imshow(markers, cmap='gray')

#Let us color boundaries in yellow. 
img[markers == -1] = [0,255,255]

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(1000) 

#Extract properties of detected regions and capture that into a pandas dataframe and save it on drive as .csv file
props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects

print(df.head())

plt.imsave('/Users/parthahazarika/Desktop/Test Data/Image/output_mask_watershed.jpg', img2, cmap='gray')
df.to_csv('/Users/parthahazarika/Desktop/Test Data/Image/output_mask_watershed.csv')