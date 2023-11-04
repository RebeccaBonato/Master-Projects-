from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras import Model, metrics
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import numpy as np

def conv_block(input_image, n_base, batch_normalization=False):
        # first convolution
        conv = Conv2D(kernel_size=3, filters=n_base, strides=(1,1), padding='same')(input_image)
        if batch_normalization:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        
        # second convolution
        conv = Conv2D(kernel_size=3, filters=n_base, strides=(1,1), padding='same')(conv)
        if batch_normalization:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        
        return conv
    
def get_unet(input_image, n_base, dropout=False, batch_normalization=False, weight_maps=False):
    
    #when adding the boundary masks after the image as input, we need to take just the first part of the input(the image)
    if weight_maps:
        e1 = conv_block(input_image[0], n_base, batch_normalization)
    else:
        e1 = conv_block(input_image, n_base, batch_normalization)
        
    e1_mp = MaxPooling2D(pool_size=(2,2))(e1)
    if dropout: 
        e1_mp = Dropout(0.2)(e1_mp)
    
    e2 = conv_block(e1_mp, n_base*2, batch_normalization)
    e2_mp = MaxPooling2D(pool_size=(2,2))(e2)
    if dropout: 
        e2_mp = Dropout(0.2)(e2_mp)
        
    e3 = conv_block(e2_mp, n_base*4, batch_normalization)
    e3_mp = MaxPooling2D(pool_size=(2,2))(e3)
    if dropout: 
        e3_mp = Dropout(0.2)(e3_mp)
    
    e4 = conv_block(e3_mp, n_base*8, batch_normalization)
    e4_mp = MaxPooling2D(pool_size=(2,2))(e4)
    if dropout: 
        e4_mp = Dropout(0.2)(e4_mp)
    
    bn = conv_block(e4_mp, n_base*16, batch_normalization)
    
    d = Conv2DTranspose(kernel_size=3, filters=8*n_base, strides=(2, 2), padding='same')(bn)
    d = Concatenate()([e4, d])
    if dropout: 
        d = Dropout(0.2)(d)
    d = conv_block(d, n_base*8, batch_normalization)
    
    d = Conv2DTranspose(kernel_size=3, filters=4*n_base, strides=(2, 2), padding='same')(d)
    d = Concatenate()([e3, d])
    if dropout: 
        d = Dropout(0.2)(d)
    d = conv_block(d, n_base*4, batch_normalization)
    
    d = Conv2DTranspose(kernel_size=3, filters=2*n_base, strides=(2, 2), padding='same')(d)
    d = Concatenate()([e2, d])
    if dropout: 
        d = Dropout(0.2)(d)
    d = conv_block(d, n_base*2, batch_normalization)
    
    d = Conv2DTranspose(kernel_size=3, filters=n_base, strides=(2, 2), padding='same')(d)
    d = Concatenate()([e1, d])
    if dropout: 
        d = Dropout(0.2)(d)
    d = conv_block(d, n_base, batch_normalization)

    #choose the right function and filter
    unet = Conv2D(1, (1, 1), activation='sigmoid') (d)

    clf = Model(inputs=input_image, outputs=unet)
    clf.summary()
    
    return clf

def dice_coef(y_true, y_pred):
    
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    
    return 1-dice_coef(y_true,y_pred)

def weighted_loss(weight_map, weight_strength):
    
    def weighted_dice_loss(y_true, y_pred):
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map)
        weight_f = weight_f * weight_strength
        weight_f = 1/(weight_f + 1)
        weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
        
        return -(2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    
    return weighted_dice_loss