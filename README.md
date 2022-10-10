# MoNuSeg
MoNuSeg- Multi-organ nuclei segmentation from H&amp;E stained histopathology image using Deep Learning

U-Net Model creation for Image Segmentation on MoNuSeg Dataset

Steps followed:

1. Data Pre-processing -

  1.1 Binary & Color mask generation from the image files and xml files provided (loop_dir.m)

  1.2 H & E normalization is done of the Images (HnE-Normal.py)

  1.3 Patching - Divide the large image and maskfiles to small patches of 256X256 (Patch_Image_Mask.py)

2. Model Creation -

  2.1 Created U-Net architecture for semantic segmentation (unet_model.py)

  2.2 Data Augmentation is done to generate copies of extra data for both image and masks (Data_Augmentation.py)

  2.3 Call the defined model, train and test it for image segmentation (Test_Model.py)

3. Data Post-processing -

  3.1 Instance segmentation is done using watershed and extract the properties of the detected regions & capture into a Pandas dataframe (watersheding.py)

4. Trained Model: (MoNuSeg_test.hdf5)

5. Test image sample: (Test_image.tif)

6. Ground Truth Mask of the sample test image: (Ground_Truth_Mask.tif)

7. Predicted Mask of the sample test image by the model: (Model_Generated_Mask.jpg)

8. Watersheding of the predicted mask of the sample image: (Model_Generated_Mask.jpg)

9. Properties of the predicted mask of the sample image: (Properties_Model_Generated_Mask.csv)
