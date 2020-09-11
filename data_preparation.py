import tarfile
import numpy as np
import h5py
import scipy.io as sio

import os
import re 
from shutil import copyfile

# Path to the downloaded file is saved
TAR_FILE_PATH = '/content/drive/My Drive/Colab Notebooks/FlowerClassification/102flowers.tgz'

# Path where you want to extract the images
EXTRACT_TO_PATH = '/content/drive/My Drive/Colab Notebooks/FlowerClassification/Untitled Folder'

# Path to the labels file is stored
LABELS_FILE_PATH = '/content/drive/My Drive/Colab Notebooks/FlowerClassification/imagelabels.mat'

IMAGE_DATA_PATH = EXTRACT_TO_PATH + '/jpg/'

TRAINING_DATA_PATH = '/content/drive/My Drive/Colab Notebooks/FlowerClassification/data/TrainingData/'

NUM_CLASSES = 102

def extract_data():
    """ Extracts the data from the tar file and saves to the destination path """

    tar = tarfile.open(TAR_FILE_PATH, 'r')
    tar.extractall(EXTRACT_TO_PATH)
    tar.close()


def create_directories_for_different_classes():
    """ Creates directories for storing images of different classes into different folders """

    for i in range(NUM_CLASSES):
        os.mkdir(TRAINING_DATA_PATH+str(i+1))



def copy_images_to_respective_folders():
    """ Copy images of different classes into different directories """

    # load lables file
    f = sio.loadmat(LABELS_FILE_PATH)
    labels = f['labels']


    ctr = 0
    for filename in os.listdir(IMAGE_DATA_PATH):
        s = re.findall(r'\d+', filename)
        image_num = int(s[0])
        label = labels[0][image_num-1]
        copyfile(IMAGE_DATA_PATH+'/'+filename, TRAINING_DATA_PATH+str(label)+'/image_'+str(image_num)+'.jpg')
        ctr = ctr + 1
        print(ctr)    

    print('Completed!!!')


if __name__ == '__main__':

    extract_data()
    create_directories_for_different_classes()
    copy_images_to_respective_folders()
