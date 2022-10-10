#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:08:13 2021

@author: parthahazarika

Transforming images and their corresponding masks together
To fit into the U-net Model

"""


from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1st Method to call
def image_agumentation(X_train, X_test, y_train, y_test):
    #Provide the same seed and keyword arguments to the fit and flow methods
    #keras.io/api/preprocessing/image/
    seed = 20
    img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(X_train, augment=True, seed=seed)

    image_generator = image_data_generator.flow(X_train, seed=seed)
    valid_img_generator = image_data_generator.flow(X_test, seed=seed)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)
    
    

    return  image_generator, mask_generator, valid_img_generator, valid_mask_generator

#2nd Method to call(two times). 1st time parameter will be image_generator & mask_generator and in 2nd time parameter will be valid_img_generator, valid_mask_generator

"""

Example:
test_image_generator = my_image_mask_generator(image_generator, mask_generator)
validation_image_generator = my_image_mask_generator(valid_img_generator, valid_mask_generator)
model.fit(test_image_generator, validation_data=validation_image_generator, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=50)

"""
def my_image_mask_generator(image_generator, mask_generator):
    #Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)
        
        
        
        
        
        
        
        
        
        