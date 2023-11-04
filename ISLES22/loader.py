import os
import numpy as np
from random import shuffle
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import tensorflow as tf
import nibabel as nib
from matplotlib import pyplot as plt
from PIL import Image
import SimpleITK as sitk
from skimage.io import imsave
from sklearn.model_selection import KFold

def Dataloader(isles_data_dir, size):

    epsilon=1e-6
    images_list = []
    masks_list = []

    for case in range (1,251):

        # Set images path.
        dwi_path = os.path.join(isles_data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %case), 'ses-0001',
                            'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" %case))
        adc_path = dwi_path.replace('dwi', 'adc')
        flair_path = dwi_path.replace('dwi', 'flair')
        mask_path = dwi_path.replace('rawdata', 'derivatives').replace('dwi', 'msk')
        
        # let's start just with a kind of image and the corrispondent mask
        # flair images have different size and number of channel
        dwi_image = nib.load(dwi_path).get_fdata()
        #adc_image = nib.load(adc_path).get_fdata()
        #flair_image = nib.load(flair_path).get_fdata()
        mask_image = nib.load(mask_path).get_fdata()

        n_slice = (mask_image.shape[2])

        # Resize and reshape if it is needed
        if dwi_image.shape[0] != 112:

            dwi_image = cv2.resize(dwi_image[:,:], (size, size), interpolation=cv2.INTER_LANCZOS4)
            dwi_image = dwi_image.reshape(size, size, n_slice)

            # Inter nearest interpolation doesn't change the value of the mask (0,1)
            mask_image = cv2.resize(mask_image[:,:], (size, size), interpolation=cv2.INTER_NEAREST)
            mask_image = mask_image.reshape(size, size, n_slice)

        # Not all the slices because we noticed that the initial ones and the final ones are completely black 
        for slice2save in range (5,n_slice-5):
            
            dwi_slice = dwi_image[:,:,slice2save]
            #normalize image slices
            dwi_slice = (dwi_slice - np.min(dwi_slice))/(np.max(dwi_slice)-np.min(dwi_slice)+epsilon)
            
            images_list.append(dwi_slice)

            mask_slice = mask_image[:,:,slice2save]
            masks_list.append(mask_slice)

    return images_list, masks_list

def Dataloader_adc(isles_data_dir, size):

    epsilon=1e-6
    images_list = []
    images2_list = []
    masks_list = []

    for case in range (1,251):

        # Set images path.
        dwi_path = os.path.join(isles_data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %case), 'ses-0001',
                            'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" %case))
        adc_path = dwi_path.replace('dwi', 'adc')
        flair_path = dwi_path.replace('dwi', 'flair')
        mask_path = dwi_path.replace('rawdata', 'derivatives').replace('dwi', 'msk')
        
        # let's start just with a kind of image and the corrispondent mask
        # flair images have different size and number of channel
        dwi_image = nib.load(dwi_path).get_fdata()
        adc_image = nib.load(adc_path).get_fdata()
        #flair_image = nib.load(flair_path).get_fdata()
        mask_image = nib.load(mask_path).get_fdata()

        n_slice = (mask_image.shape[2])

        # Resize and reshape if it is needed
        if dwi_image.shape[0] != 112:

            dwi_image = cv2.resize(dwi_image[:,:], (size, size), interpolation=cv2.INTER_LANCZOS4)
            dwi_image = dwi_image.reshape(size, size, n_slice)

            # Inter nearest interpolation doesn't change the value of the mask (0,1)
            mask_image = cv2.resize(mask_image[:,:], (size, size), interpolation=cv2.INTER_NEAREST)
            mask_image = mask_image.reshape(size, size, n_slice)
            
            adc_image = cv2.resize(adc_image[:,:], (size, size), interpolation=cv2.INTER_LANCZOS4)
            adc_image = adc_image.reshape(size, size, n_slice)

        # Not all the slices because we noticed that the initial ones and the final ones are completely black
        for slice2save in range (5,n_slice-5):
            
            dwi_slice = dwi_image[:,:,slice2save]
            #normalize image slices
            dwi_slice = (dwi_slice - np.min(dwi_slice))/(np.max(dwi_slice)-np.min(dwi_slice)+epsilon)
            
            images_list.append(dwi_slice)
            
            adc_slice = adc_image[:,:,slice2save]
            #normalize image slices
            adc_slice = (adc_slice - np.min(adc_slice))/(np.max(adc_slice)-np.min(adc_slice)+epsilon)
            
            images2_list.append(adc_slice)

            mask_slice = mask_image[:,:,slice2save]
            masks_list.append(mask_slice)

    return images_list, images2_list, masks_list

def Dataloader_boundary(isles_data_dir, size):

    epsilon=1e-6
    images_list = []
    masks_list = []
    boundaries_list = []

    for case in range (1,251):

        # Set images path.
        dwi_path = os.path.join(isles_data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %case), 'ses-0001',
                            'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" %case))
        adc_path = dwi_path.replace('dwi', 'adc')
        flair_path = dwi_path.replace('dwi', 'flair')
        mask_path = dwi_path.replace('rawdata', 'derivatives').replace('dwi', 'msk')
        
         
        # let's start just with a kind of image and the corrispondent mask
        # flair images have different size and number of channel
        dwi_image = nib.load(dwi_path).get_fdata()
        #adc_image = nib.load(adc_path).get_fdata()
        #flair_image = nib.load(flair_path).get_fdata()
        mask_image = nib.load(mask_path).get_fdata()

        n_slice = (mask_image.shape[2])

        # Resize and reshape if it is needed
        if dwi_image.shape[0] != 112:
            
            dwi_image = cv2.resize(dwi_image[:,:], (size, size), interpolation=cv2.INTER_LANCZOS4)
            dwi_image = dwi_image.reshape(size, size, n_slice)

            # Inter nearest interpolation doesn't change the value of the mask (0,1)
            mask_image = cv2.resize(mask_image[:,:], (size, size), interpolation=cv2.INTER_NEAREST)
            mask_image = mask_image.reshape(size, size, n_slice)

        # Not all the slices because we noticed that the initial ones and the final ones are completely black 
        for slice2save in range (5,n_slice-5):
            
            dwi_slice = dwi_image[:,:,slice2save] 
            #normalize image slices
            dwi_slice = (dwi_slice - np.min(dwi_slice))/(np.max(dwi_slice)-np.min(dwi_slice)+epsilon)
            images_list.append(dwi_slice)

            mask_slice = mask_image[:,:,slice2save]
            masks_list.append(mask_slice)
        
            #create boundary mask
            kernel = np.ones((2, 2), np.uint8)
            boundary = cv2.morphologyEx(mask_slice, cv2.MORPH_GRADIENT, kernel)
            boundaries_list.append(boundary)

    return images_list, masks_list, boundaries_list


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def get_train_val_list(Image_list, Mask_list, Train_Percent):
    
    combined = list(zip(Image_list, Mask_list))
    random.shuffle(combined)
    Image_list[:], Mask_list[:] = zip(*combined)
    
    train_split = int(Train_Percent*len(Image_list))
    
    train_list_image = Image_list[:train_split]
    train_list_mask = Mask_list[:train_split]
    val_list_image = Image_list[train_split:]
    val_list_mask = Mask_list[train_split:]
    
    
    return train_list_image, train_list_mask, val_list_image, val_list_mask


def get_train_val_list_with_boundaries(Image_list, Mask_list, Boundary_list, Train_Percent):
    
    combined = list(zip(Image_list, Mask_list, Boundary_list))
    random.shuffle(combined)
    Image_list[:], Mask_list[:], Boundary_list[:] = zip(*combined)
    
    train_split = int(Train_Percent*len(Image_list))
    
    train_list_image = Image_list[:train_split]
    train_list_mask = Mask_list[:train_split]
    train_list_boundary = Boundary_list[:train_split]
    val_list_image = Image_list[train_split:]
    val_list_mask = Mask_list[train_split:]
    val_list_boundary = Boundary_list[train_split:]
    
    
    return train_list_image, train_list_mask, train_list_boundary, val_list_image, val_list_mask, val_list_boundary

def get_train_val_list_with_adc(Image_list, Mask_list, adc_list, Train_Percent):
    
    combined = list(zip(Image_list, Mask_list, adc_list))
    random.shuffle(combined)
    Image_list[:], Mask_list[:], adc_list[:] = zip(*combined)
    
    train_split = int(Train_Percent*len(Image_list))
    
    train_list_image = Image_list[:train_split]
    train_list_mask = Mask_list[:train_split]
    train_list_adc = adc_list[:train_split]
    val_list_image = Image_list[train_split:]
    val_list_mask = Mask_list[train_split:]
    val_list_adc = adc_list[train_split:]
    
    
    return train_list_image, train_list_mask, train_list_adc, val_list_image, val_list_mask, val_list_adc


def split_sets(num_folds, inputs, targets, shuffle=True):
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=shuffle)
    splitted_indeces = kfold.split(inputs, targets)
    
    return splitted_indeces


def generator(x_train, y_train, batch_size, sample_weight):
    
    n_train_sample = len(x_train)
    while True:
               
        for ind in (range(0, n_train_sample, batch_size)):
            
            batch_img = x_train[ind:ind+batch_size]
            batch_label = y_train[ind:ind+batch_size]
            
            # Sanity check assures batch size always satisfied
            # by repeating the last 2-3 images at last batch.
            length = len(batch_img)
            if length == batch_size:
                pass
            else:
                for tmp in range(batch_size - length):
                    batch_img = np.append(batch_img, np.expand_dims(batch_img[-1],axis=0), axis = 0)
                    batch_label = np.append(batch_label, np.expand_dims(batch_label[-1], axis=0), axis = 0)
        
            backgound_value = x_train.min()
            data_gen_args = dict(rotation_range=0.1, cval = backgound_value, horizontal_flip = True)
            
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            image_generator = image_datagen.flow(batch_img, shuffle=False,
                                                 batch_size=batch_size,
                                                 seed=1)
            
            #try to solve the unbalanced problem increasing the weight of white pixels
            sample_weight[batch_label==0.] = 1.
            sample_weight[batch_label==1.] = 250.
            
            mask_generator = mask_datagen.flow(batch_label, shuffle=False,
                                               batch_size=batch_size,
                                               seed=1, sample_weight=sample_weight)
            
            image = image_generator.next()
            label = mask_generator.next()
            
            yield image, label
            

def combine_generator(gen1, gen2, gen3=None):
    
    while True:
        if gen3 is None:
            yield(gen1.next(), gen2.next())
        else:
            x = gen1.next()
            y = gen2.next()
            w = gen3.next()
            yield([x, w], y)

def generator_with_weights(x_train, y_train, weights_train, batch_size, sample_weight):
    
    backgound_value = x_train.min()
    data_gen_args = dict(rotation_range=0.1,
                         #width_shift_range=0.1,
                         #height_shift_range=0.1,
                         cval=backgound_value,
                         #zoom_range=0.2,
                         horizontal_flip=True)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)
    

    image_generator = image_datagen.flow(x_train, shuffle=False,
                                         batch_size=batch_size,
                                         seed=1)
    
    #try to solve the unbalanced problem increasing the weight of white pixels
    sample_weight[y_train==0.] = 1.
    sample_weight[y_train==1.] = 250.
    
    mask_generator = mask_datagen.flow(y_train, shuffle=False,
                                       batch_size=batch_size,
                                       seed=1, sample_weight=sample_weight)

    weight_generator = weights_datagen.flow(weights_train, shuffle=False,
                                         batch_size=batch_size,
                                         seed=1)

    train_generator = combine_generator(image_generator, mask_generator, weight_generator)

    return train_generator


def get_autocontext_fold(y_pred, fold_no, num_folds, length_list):

    autocontext_val = y_pred[length_list[fold_no]:length_list[fold_no+1]]
    autocontext_train= []
    
    if fold_no!=0:
        autocontext_train.append(y_pred[0:length_list[fold_no]])
        
    if fold_no!=(num_folds-1):
        autocontext_train.append(y_pred[length_list[fold_no+1]:])
    
    #change list in array
    if len(autocontext_train)==1:
        autocontext_train=autocontext_train[0]
        
    elif len(autocontext_train)==2:
        autocontext_train=np.concatenate((autocontext_train[0], autocontext_train[1]))
                
    return autocontext_train, autocontext_val

def save_data(clf_hist, fold_no, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall):
    
    loss[fold_no-1,:] = clf_hist.history['loss']
    val_loss[fold_no-1,:] = clf_hist.history['val_loss']
    dice_coef[fold_no-1,:] = clf_hist.history['dice_coef']
    val_dice_coef[fold_no-1,:] = clf_hist.history['val_dice_coef']
    precision[fold_no-1,:] = clf_hist.history['precision']
    val_precision[fold_no-1,:] = clf_hist.history['val_precision']
    recall[fold_no-1,:] = clf_hist.history['recall']
    val_recall[fold_no-1,:] = clf_hist.history['val_recall']
    
def save_data_autocontext(clf_hist, fold_no, step, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall):
    
    loss[fold_no,step-1,:] = clf_hist.history['loss']
    val_loss[fold_no,step-1,:] = clf_hist.history['val_loss']
    dice_coef[fold_no,step-1,:] = clf_hist.history['dice_coef']
    val_dice_coef[fold_no,step-1,:] = clf_hist.history['val_dice_coef']
    precision[fold_no,step-1,:] = clf_hist.history['precision']
    val_precision[fold_no,step-1,:] = clf_hist.history['val_precision']
    recall[fold_no,step-1,:] = clf_hist.history['recall']
    val_recall[fold_no,step-1,:] = clf_hist.history['val_recall']

#can be used with both list boundaries and adc
def expand_dimension2(train_list_image, train_list_mask, train_list_boundary, val_list_image, val_list_mask, val_list_boundary):
    
    train_images=np.expand_dims(np.array(train_list_image), axis=3)
    train_masks=np.expand_dims(np.array(train_list_mask), axis=3)
    val_images=np.expand_dims(np.array(val_list_image), axis=3)
    val_masks=np.expand_dims(np.array(val_list_mask), axis=3)
    train_boundaries = np.expand_dims(np.array(train_list_boundary), axis=3)
    val_boundaries = np.expand_dims(np.array(val_list_boundary), axis=3)
    
    del train_list_image, train_list_mask, val_list_image, val_list_mask, train_list_boundary, val_list_boundary
    
    return train_images, train_masks, train_boundaries, val_images, val_masks, val_boundaries

    
def expand_dimension(train_list_image, train_list_mask, val_list_image, val_list_mask):
    
    train_images=np.expand_dims(np.array(train_list_image), axis=3)
    train_masks=np.expand_dims(np.array(train_list_mask), axis=3)
    val_images=np.expand_dims(np.array(val_list_image), axis=3)
    val_masks=np.expand_dims(np.array(val_list_mask), axis=3)
    
    del train_list_image, train_list_mask, val_list_image, val_list_mask
    
    return train_images, train_masks, val_images, val_masks