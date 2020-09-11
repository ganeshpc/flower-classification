# flower-classification
Classifying the 102 types of the flowers using mobilenet v2 and transfer learning.

## Flower image classification using costom dataset

##### Download Data
In this project [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) is used. You can download the [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and the [label file](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat).
Dataset is in the form of tar file and once extracted, all the images are inside the jpg folder.
The labels file contains a array with index being the number of the image and the number at the index reffering to the class to which the image belongs.

##### Keras Data Generator
Here keras image generator is used to feed data to the neural network. The data should be structured in the following manner:
![Folder structure](https://miro.medium.com/max/700/0*wl6rLXC0wNL27fnd.png).

In the above image the training_images is the subfolder which contains multiple directories and each directory contains images belonging to one perticular class.

##### Dataset preparation
As the keras data generator requires training data to be in perticular structure we need to prepare the dataset according the needs. For preparing dataset refer to the [this file](), which converts the data in above mentioned format.

In [data_preparation.py](https://github.com/ganeshpc/flower-classification/blob/master/data_preparation.py) file you need to change the following variables:

> TAR_FILE_PATH = ' '    #provide path to the downloaded tar file
> EXTRACT_TO_PATH = ' '  #path where you want to extract the tar file
> LABELS_FILE_PATH = ' '  #path where labels file is located
> TRAINING_DATA_PATH = ' ' #path where you want to save prepared dataset (eg: folder named training_images in above image)

And run the file data_preparation.py

##### Creating and Training Model
To feed the data to the model we will use keras data generators. 
This model is not created from the scratch rather a pretrained model for generating features is used and we add classification layer at the end. The pretrained feature extracted model is from tensorflow hub. The pretrained model is [MobileNet v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4), it is lite model optimized to be used on smaller devices such as raspberry pi.
So we load the pretrained model as a keras layer and attach classification layers in front of it and while training the previous layers are frozen. This type of learning is called **Transfer Learning.**

![transfer learning](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F2000%2F1*f2_PnaPgA9iC5bpQaTroRw.png&f=1&nofb=1)

Code to the model training can be found in [model.py](https://github.com/ganeshpc/flower-classification/blob/master/model.py).

##### UI for classification
Here simple desktop application is created using tkinter python library to classify the images. To classify the images run [Classification_GUI.py](https://github.com/ganeshpc/flower-classification/blob/master/Classification_GUI.py)
![Classification_UI](https://github.com/ganeshpc/flower-classification/blob/master/resources/Classification_UI.PNG?raw=true)
