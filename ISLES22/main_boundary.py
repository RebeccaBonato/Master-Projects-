from model import conv_block, get_unet, weighted_loss, dice_coef
from loader import Dataloader_boundary,get_train_val_list_with_boundaries, generator_with_weights,expand_dimension2
from plot import plot_simple
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision ,Recall

%matplotlib inline

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Define Parameters
isles_data_dir = 'dataset-ISLES22/'
size = 112
Train_Percent=0.7
n_base=8
LR = 1e-4
Batch_size = 32
n_epoch = 200
weight_maps=True
dropout=True
batch_normalization=True

optim = Adam
Metric = [dice_coef, Precision(), Recall()]
weight_strength=1

images, masks, boundaries = Dataloader_boundary(isles_data_dir, size)
train_list_image, train_list_mask, train_list_boundary, val_list_image, val_list_mask, val_list_boundary = get_train_val_list_with_boundaries(images,
                                                                                                          masks, boundaries, Train_Percent)

train_images, train_masks, train_boundaries, val_images, val_masks, val_boundaries = expand_dimension2(train_list_image,
                                                                                    train_list_mask,
                                                                                    train_list_boundary,
                                                                                    val_list_image,
                                                                                    val_list_mask,
                                                                                    val_list_boundary)

#the number of black pixels is 250 times the number of white pixels
sample_weights = np.zeros(shape=train_masks.shape)

input_shape = (size,size,1)
input_image = Input(input_shape, name ='input_layer')
loss_weights = Input(shape = (size,size,1))

clf = get_unet([input_image, loss_weights], n_base, dropout, batch_normalization, weight_maps)

clf.compile(loss=weighted_loss(loss_weights,weight_strength),
            optimizer=optim(learning_rate=LR),
            metrics=Metric)

steps_per_epoch = train_images.shape[0]//Batch_size
    
train_generator = generator_with_weights(train_images, train_masks, train_boundaries, Batch_size, sample_weights)
clf_hist = clf.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs = n_epoch,
        validation_steps = len(val_images),
        validation_data=([val_images, val_boundaries], val_masks),
        shuffle = True)

plot_simple(clf_hist)