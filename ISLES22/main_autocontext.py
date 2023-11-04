from model import conv_block, get_unet, dice_coef, dice_coef_loss
from loader import Dataloader, get_train_val_list, generator, get_autocontext_fold, split_sets, save_data_autocontext
from plot import plot_autocontext
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
n_epoch = 150
dropout=True
batch_normalization=True

step=2
output_path = 'autocontext/'
num_folds = 3

optim = Adam
f_loss = dice_coef_loss
Metric = [dice_coef, Precision(), Recall()]

loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall = (np.zeros((num_folds, step, n_epoch)) for i in range(8))

image_list, mask_list = Dataloader(isles_data_dir, size)

images=np.expand_dims(np.array(image_list), axis=3)
masks=np.expand_dims(np.array(mask_list), axis=3)
del image_list, mask_list

#the number of black pixels is 250 times the number of white pixels
sample_weights = np.zeros(shape=(Batch_size,size,size,1))

input_shape = (size,size,2)
input_image = Input(input_shape, name ='input_layer')

model_predictions=np.zeros((len(images), size,size,1))

splitted_indeces = split_sets(num_folds, images, masks)

#save indeces in order to do more steps
train_list=[]
test_list=[]
for train, test in splitted_indeces:
    train_list.append(train)
    test_list.append(test)

for s in range (step):
    
    length_list = [0]
    length = 0
    
    for fold_no in range(num_folds):

        #since different folds can have a different number of images/masks, we keep track of their length
        length += len(test_list[fold_no])
        length_list.append(length)

        if s==0:
            autocontext_train = np.zeros_like(images[train_list[fold_no]]) + 0.5
            train_images = np.concatenate((images[train_list[fold_no]], autocontext_train), axis=-1)
            autocontext_val = np.zeros_like(images[test_list[fold_no]]) + 0.5
            val_images = np.concatenate((images[test_list[fold_no]], autocontext_val), axis=-1)

        else:

            y_pred = np.load(output_path + 'step' + str(s-1) + '.npy')
            autocontext_train, autocontext_val = get_autocontext_fold(y_pred, fold_no, num_folds, length_list)
            train_images = np.concatenate((images[train_list[fold_no]], autocontext_train), axis=-1)
            val_images = np.concatenate((images[test_list[fold_no]], autocontext_val), axis=-1)
        
        clf = get_unet(input_image, n_base, dropout, batch_normalization)
        clf.compile(loss=f_loss,
                 optimizer=optim(learning_rate=LR),
                 metrics=Metric)
        
        steps_per_epoch = train_images.shape[0]//Batch_size

        train_generator = generator(train_images, masks[train_list[fold_no]], Batch_size, sample_weights)
        clf_hist = clf.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs = n_epoch, validation_steps = len(val_images),
            validation_data = (val_images, masks[test_list[fold_no]]),
            shuffle = True)

        save_data_autocontext(clf_hist, fold_no, step, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall)

        model_predictions[length_list[fold_no]:length_list[fold_no+1]] = clf.predict(val_images, batch_size=1)
        np.save(output_path + 'step' + str(s) + '.npy', model_predictions)
        
        del clf
        
plot_autocontext(step, num_folds, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall)