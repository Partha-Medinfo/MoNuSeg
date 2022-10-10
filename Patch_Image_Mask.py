#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 11 13:00:44 2021

@author: parthahazarika

This program will extract pathces from large images and masks to train the U-Net model and save in the Drive

"""


from patchify import patchify
import tifffile as tiff
import cv2
import os

#The below function will create non overlapping 256X256 patches from a large image

def making_patchs_image(image_path):
    
    filename = os.path.basename(image_path)
    new_filename = filename.replace('.tiff', '')
    large_image = cv2.imread(image_path,0)
    
    #print (large_image)
    
    patches_img = patchify(large_image, (256, 256), step=256) # No overalpping 
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('/Users/parthahazarika/Desktop/Patch-Generatation/Patch/Image/' + str(new_filename) + '_' + str(i)+str(j)+ ".tif", single_patch_img)
            

#The below function will create non overlapping 256X256 patches from a large mask
            
            
def making_patchs_mask(mask_path):
    
    filename = os.path.basename(mask_path)
    new_filename = filename.replace('.tif', '')
    large_mask = cv2.imread(mask_path,0)

    
    #print (large_mask)
    
    patches_mask = patchify(large_mask, (256, 256), step=256)  
    
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite('/Users/parthahazarika/Desktop/Patch-Generatation/Patch/Mask/' + str(new_filename) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            single_patch_mask = single_patch_mask / 255.
              

"""

Upon providing the detailed folder name of both Images and Masks the below lines of code will traverse the directory struc-
ture and and will call both the methods for Patching images and masks 

"""

directory_image = '/Users/parthahazarika/Desktop/Patch-Generatation/Image'
directory_mask = '/Users/parthahazarika/Desktop/Patch-Generatation/Mask'

for filename in os.listdir(directory_image):
    if filename.endswith(".tiff"):
    #if filename.endswith(".tif"):
        newfilename = os.path.join(directory_image, filename)
        print(newfilename)
        making_patchs_image(newfilename)
        
for filename in os.listdir(directory_mask):
    if filename.endswith(".tif"):
        newfilename = os.path.join(directory_mask, filename)
        print(newfilename)     
        making_patchs_mask(newfilename)
        
        
        