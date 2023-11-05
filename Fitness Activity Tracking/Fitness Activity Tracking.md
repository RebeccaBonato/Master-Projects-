# Fitness Activity Tracking

The aim of the project was to create a machine or deep learning algorithm capable of classifying fitness videos into the correct type of exercise performed. Each video contained more than one exercise, and for this reason we were provided with a file detailing when each exercise began and ended in order to have a dataset with appropriate labels. 

<div>
  <img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/5minex.png" alt="Immagine 1" width="300" />
  <img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/7minex.png" alt="Immagine 2" width="300" />
</div>

The challenges we faced in carrying out this project were many. 
   * First of all, each video was characterised by a series of frames of different size and the number of frames per exercise was variable.
   * The resolution changed from video to video.
   * The videos were very heavy and therefore difficult to process by a neural network with the resources we had available.
   * The camera was not always fixed on the subject but sometimes moved around it. The video sometimes also framed people who were not taking part in the exercise.
   * Finally, the people performing the exercises were both professionals and amateurs. In some cases, the exercise was not performed correctly and therefore difficult for the neural network to classify. 

The first step taken was a [data cleaning operation](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/1_Annotations_to_dataframe.ipynb), both manual and automatic, to identify any errors. This made it much easier to process data. 

We soon realised that working with videos was difficult for the resources we had available. Consequently, using an already trained network, [Movenet](https://www.tensorflow.org/hub/tutorials/movenet?hl=fr), we extracted the muscle joints of the subject performing the exercise for each frame ([here the code](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/2_Create_lighter_dataset.ipynb)). Once this was done, each video was reduced to 34 signals over time: in fact, there were 17 joint joints extracted and each joint was characterised by two coordinates (x and y). This made the dataset much lighter. 

At this point, the initial dataset was divided into training and validation in order to make it as balanced as possible. In addition, hot encoding was applied for the labels. The network has been built: a combination of a 1D convolutional neural network and a LSTM is used to do the classification of the activity based on the coordinates of the joints. Categorical crossentropy has been used as a loss function while accuracy is the metric used to evaluate performance along with confusion matrix. Finally, the best model was saved and adopted on a new dataset, which was not used for either training or validation. The code related to the model and its evaluation can be found [here](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/3_Network.ipynb) 

More details and our results can be found in our [final report](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/Fitness_Activity_Tracking_.pdf). 
This project was carried out at the KTH Royal Institute of Technology, as the final project of the course on machine learning applied to sport and health. This project was carried out in pairs of students; in my case, the collaboration with Benjamin Darcòt was crucial to the success of the project. 


[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
