# Dog_Emotions

Link to the dataset: https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction
Built a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to identify emotions in images of dogs.
The model architecture includes several convolutional layers followed by max pooling and dense layers. 
The input shape of the images is defined as 224 x 224 pixels with 3 color channels (RGB).

The model is compiled with the Adam optimizer and categorical cross-entropy loss function. The accuracy metric is also included.

An ImageDataGenerator is created to perform data augmentation on the training data, including rotation, shift, shear, zoom, and flip. 
The training data is loaded from a directory using the flow_from_directory method. 
Similarly, a separate ImageDataGenerator is created for preprocessing the validation data, which is also loaded from a directory.

The model is trained using the fit method, with the training data and validation data as inputs for a total of 10 epochs.

Finally, the trained model is saved as a h5 file with the name "dog_emotions_model.h5".

Below is the link to the video walkthrough of the whole assignment:
https://drive.google.com/drive/folders/1GkMs-7p0T70oaeGqU-PE7qM07snQT7v1?usp=sharing
