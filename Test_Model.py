#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:26:51 2021

@author: parthahazarika

"""

from unet_model import unet_model
#from Data_Augmentation import image_agumentation, my_image_mask_generator
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import random
from matplotlib import pyplot as plt

image_directory = '/Users/parthahazarika/Desktop/Patch-Generatation/Patch/Image/'
mask_directory = '/Users/parthahazarika/Desktop/Patch-Generatation/Patch/Mask/'

SIZE = 256
image_dataset = []  #Pandas can be used but here I am using a list format.  
mask_dataset = []  

#print("check")

filenames = os.listdir(image_directory)
#loading images in the right order, otherwise it will load randomly 
filenames.sort() 
#i = 1  
for filename in filenames:
    if filename.endswith(".tif"):
        #print(i,filename)
        #i = i+1
        full_path = image_directory + filename
        #print(i,full_path)
        #i=i+1
        image = cv2.imread(full_path,0)
        #print(image)
        image = Image.fromarray(image)
        #print(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
        #print(len(image_dataset))
 
masks = os.listdir(mask_directory)
#loading masks in the right order, otherwise it will load randomly
masks.sort()
i = 1
for mask in masks:
    if mask.endswith(".tif"):
        #print(i, mask)
        #i = i+1
        mask_full_path = mask_directory + mask
        #print(i, mask_full_path)
        #i = i + 1
        image = cv2.imread(mask_full_path, 0)
        image = Image.fromarray(image)
        #print(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
        #print(len(mask_dataset))
        
        
#Image normalization
image_dataset = np.expand_dims(tf.keras.utils.normalize(np.array(image_dataset), axis=1),3)
#Rescale mask - 0 to 1
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

from sklearn.model_selection import train_test_split
#print("check")     
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0) 

#print(len(X_train)) 
#print(len(y_train)) 
#print(len(X_test)) 
#print(len(y_test)) 

"""
#Randomly checking whether the images and the corresponding masks are aligned properly or not

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()   
"""    
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]  

def get_model():
    return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

"""

##### If Data Augmentation Part is used to generate more images and their corresponding masks, below code will do that #####

image_generator, mask_generator, valid_img_generator, valid_mask_generator = image_agumentation(X_train, X_test, y_train, y_test)
test_image_generator = my_image_mask_generator(image_generator, mask_generator)
validation_image_generator = my_image_mask_generator(valid_img_generator, valid_mask_generator)

batch_size = 10
steps_per_epoch = 3*(len(X_train))//batch_size

model.fit(test_image_generator, validation_data=validation_image_generator, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=50)


"""


train = model.fit(X_train, y_train, 
                    batch_size = 10, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('/Users/parthahazarika/Desktop/SEG-MODEL/MoNuSeg_test.hdf5')

#Check the accuracy of the model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#IoU (Intersection Over Union)
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5 # Anything >0.5, make it 1 and anything <0.5 make it 0

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre for this model is: ", iou_score)

#Test the model
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.imsave('/Users/parthahazarika/Desktop/Data/Dataoutput1.tiff', prediction, cmap='gray')
plt.imsave('/Users/parthahazarika/Desktop/Data/test1.tiff', test_img[:, :, 0], cmap='gray')
plt.imsave('/Users/parthahazarika/Desktop/Data/mask1.jpg', prediction, cmap='gray')

#Test a different image i.e, an external image 
test_img_dif = cv2.imread('/Users/parthahazarika/Desktop/Test Data/Image/TCGA-B0-5698-01Z-00-DX1-HnE-Normalized_01.tif', 0)
test_img_dif_norm = np.expand_dims(tf.keras.utils.normalize(np.array(test_img_dif), axis=1),2)
test_img_dif_norm=test_img_dif_norm[:,:,0][:,:,None]
test_img_dif_input=np.expand_dims(test_img_dif_norm, 0)

#Predict and threshold for values above 0.5 probability
prediction_dif_image = (model.predict(test_img_dif_input)[0,:,:,0] > 0.5).astype(np.uint8)
plt.imsave('/Users/parthahazarika/Desktop/Test Data/Image/output_mask.jpg', prediction_dif_image, cmap='gray')
#plt.imsave('/Users/parthahazarika/Desktop/Test Data/Image/output_mask.tiff', prediction_dif_image, cmap='gray')
