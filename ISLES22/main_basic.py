from model import conv_block, get_unet, dice_coef, dice_coef_loss
from loader import Dataloader, get_train_val_list, generator, split_sets, expand_dimension, save_data
from plot import plot_Kfold, plot_simple
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision ,Recall

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

%matplotlib inline

# Define Parameters
isles_data_dir = 'dataset-ISLES22/'
size = 112

n_base=8
LR = 1e-4
Batch_size = 32
n_epoch = 200
dropout=True
batch_normalization=True

kfold=False
Train_Percent=0.7
num_folds=3

optim = Adam
f_loss = dice_coef_loss
#f_loss = 'binary_crossentropy'
Metric = [dice_coef, Precision(), Recall()]

images, masks = Dataloader(isles_data_dir, size)

#the number of black pixels is 250 times the number of white pixels
sample_weights = np.zeros(shape=(Batch_size,size,size,1))

input_shape = (size,size,1)
input_image = Input(input_shape, name ='input_layer')

if kfold:  

    loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall = (np.zeros((num_folds, n_epoch)) for i in range(8))

    images=np.expand_dims(np.array(images), axis=3)
    masks=np.expand_dims(np.array(masks), axis=3)
    
    splitted_indeces = split_sets(num_folds, images, masks)

    fold_no = 1
    for train, test in splitted_indeces:

        clf = get_unet(input_image, n_base, dropout, batch_normalization)

        clf.compile(loss=f_loss,
                 optimizer=optim(learning_rate=LR),
                 metrics=Metric)

        steps_per_epoch = images[train].shape[0]//Batch_size
        print(f'Training for fold {fold_no} ...')

        train_generator = generator(images[train], masks[train], Batch_size, sample_weights)
        clf_hist = clf.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs = n_epoch,
                validation_steps = len(images[test]),
                validation_data=(images[test], masks[test]),
                shuffle = True)

        save_data(clf_hist, fold_no, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall)

        fold_no = fold_no + 1
        del clf
        
    plot_Kfold(loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall)

else: 

    train_list_image, train_list_mask, val_list_image, val_list_mask = get_train_val_list(images, masks, Train_Percent)

    train_images, train_masks, val_images, val_masks = expand_dimension(train_list_image,
                                                       train_list_mask, val_list_image, val_list_mask)
    
    clf = get_unet(input_image, n_base, dropout, batch_normalization)

    clf.compile(loss=f_loss,
         optimizer=optim(learning_rate=LR),
         metrics=Metric)

    steps_per_epoch = train_images.shape[0]//Batch_size

    train_generator = generator(train_images, train_masks, Batch_size, sample_weights)
    clf_hist = clf.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs = n_epoch,
        validation_steps = len(val_images),
        validation_data=(val_images, val_masks),
        shuffle = True)
    
    plot_simple(clf_hist)