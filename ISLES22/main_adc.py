from model import conv_block, get_unet, dice_coef, dice_coef_loss
from loader import Dataloader_adc, get_train_val_list_with_adc, generator, expand_dimension2, save_data
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

n_base=8
LR = 1e-4
Batch_size = 32
n_epoch = 200
dropout=True
batch_normalization=True

Train_Percent=0.7

optim = Adam
f_loss = dice_coef_loss
#f_loss = 'binary_crossentropy'
Metric = [dice_coef, Precision(), Recall()]

images, adc, masks = Dataloader_adc(isles_data_dir, size)

#the number of black pixels is 250 times the number of white pixels
sample_weights = np.zeros(shape=(Batch_size,size,size,1))

input_shape = (size,size,2)
input_image = Input(input_shape, name ='input_layer')

train_list_image, train_list_mask, train_list_adc, val_list_image, val_list_mask, val_list_adc = get_train_val_list_with_adc(images, 
                                                                                masks, adc, Train_Percent)

train_images, train_masks, train_adc, val_images, val_masks, val_adc = expand_dimension2(train_list_image,
                                                   train_list_mask, train_list_adc, val_list_image,
                                                    val_list_mask, val_list_adc)

train_input = np.concatenate((train_images, train_adc), axis=-1)
test_input = np.concatenate((val_images, val_adc), axis=-1)

clf = get_unet(input_image, n_base, dropout, batch_normalization)

clf.compile(loss=f_loss,
     optimizer=optim(learning_rate=LR),
     metrics=Metric)

steps_per_epoch = train_input.shape[0]//Batch_size

train_generator = generator(train_input, train_masks, Batch_size, sample_weights)
clf_hist = clf.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs = n_epoch,
    validation_steps = len(test_input),
    validation_data=(test_input, val_masks),
    shuffle = True)

plot_simple(clf_hist)