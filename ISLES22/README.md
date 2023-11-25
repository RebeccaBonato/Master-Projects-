# ISLES 2022

## *Deep learning in medical image analysis*

## Aim
The goal of the Ischemic Stroke Lesion Segmentation Challenge (ISLES'22) is to evaluate the best networks capable of segment stroke lesion MR images. These models are important in order to evaluate patients' disease outcome and to define optimal treatment and rehabilitation strategies. The data-set of the challenge includes acute to sub-acute infarct lesions, with high variability in size, quantity and location.

Here are two examples of DWI images and their mask.

![dwi_mask00093](https://user-images.githubusercontent.com/92582518/204902293-ec70d5d6-55d2-43a9-a4c8-fa03a4560039.jpg)
![dwi_image00093](https://user-images.githubusercontent.com/92582518/204905838-247b0828-941a-491d-9e82-56887732d263.jpg)

![dwi_mask00037](https://user-images.githubusercontent.com/92582518/204902514-1ea06149-d7f2-4295-8242-963ebfab32ed.jpg)
![dwi_image00037](https://user-images.githubusercontent.com/92582518/204902530-477d8cec-9ff8-4061-9e21-f942b653e796.jpg)

## :handshake: Contributors
Bonato Rebecca and Scoglio Marianne

This is a private school project.

## Project description and results
A U-Net was implemented and trained with DWI images (ADC were adedd later) of stroke lesions and their ground-truth. Various techniques were used to improve and evaluate the model: tuning of hyperparameters (including class weights), autocontext, K-fold cross validation, data augmentation and weighted loss for boundary masks. All the knowledge comes from the previous Labs and from researches on internet.

The best model we obtained can be run in main_basic.py (or in main_adc.py since they converge to the same values). It reaches a performance with a dice coefficient of 0.834, a precision of 0.910 and a recall of 0.860.

## :card_file_box: Files and contents

The final project related files are under 'ISLES22'.
Since we used different methods to find the best models, we organized them in different types of main.py. Here you can find the the folder with the [final report](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/text/ISLES2022.pdf) containing the best results achieved. In the appendix other attempts are organized in a table. 

[Loader.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/loader.py): This file contains the function used to load, save, divide, and modify the images.

[Model.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/model.py): This file contains the U-Net and the loss functions.

[Plot.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/plot.py): This file contains the functions to plot the results.

[Main_basic.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/main_basic.py): This file allows to run the basic model.

[Main_boundary.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/main_boundary.py): This file allows to run the model using weighted dice coefficient loss for boundary masks.

[Main_autocontext.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/main_autocontext.py): This file allows to run the model using the autocontext.

[Main_adc.py](https://github.com/RebeccaBonato/Master-Projects-/blob/main/ISLES22/main_adc.py): This file allows to run the model adding ADC images.

[dataset-ISLES22](https://github.com/RebeccaBonato/Master-Projects-/tree/main/ISLES22/dataset-ISLES22): This folder contains the data-set.

[autocontext](https://github.com/RebeccaBonato/Master-Projects-/tree/main/ISLES22/autocontext): This folder is needed to save the data during autocontext.





[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
