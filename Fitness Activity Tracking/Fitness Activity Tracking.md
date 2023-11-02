# Fitness Activity Tracking

The aim of the project was to create a machine or deep learning algorithm capable of classifying fitness videos into the correct type of exercise performed. Each video contained more than one exercise, and for this reason we were provided with a file detailing when each exercise began and ended in order to have a dataset with appropriate labels. 

The challenges we faced in carrying out this project were many. 
   * First of all, each video was characterised by a series of frames of different size and the number of frames per exercise was variable.
   * The resolution changed from video to video.
   * The videos were very heavy and therefore difficult to process by a neural network with the resources we had available.
   * The camera was not always fixed on the subject but sometimes moved around it. The video sometimes also framed people who were not taking part in the exercise.
   * Finally, the people performing the exercises were both professionals and amateurs. In some cases, the exercise was not performed correctly and therefore difficult for the neural network to classify. 

The first step taken was a [data cleaning operation](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/1_Annotations_to_dataframe.ipynb), both manual and automatic, to identify any errors. This made it much easier to process data. 

We soon realised that working with videos was difficult for the resources we had available. Consequently, using an already trained network, [Movenet](https://www.tensorflow.org/hub/tutorials/movenet?hl=fr), we extracted the muscle joints of the subject performing the exercise for each frame ([here the code](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Fitness%20Activity%20Tracking/2_Create_lighter_dataset.ipynb)). In this way, we found ourselves working with signals: the movement of each right muscle over time. This made the dataset much lighter. 





This project was carried out at the KTH Royal Institute of Technology, as the final project of the course on machine learning applied to sport and health. This project was carried out in pairs of students; in my case, the collaboration with Benjamin Darcòt was crucial to the success of the project. 


[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
