import keras
from keras import layers
from keras.layers import Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import tensorflow_hub as hub

import cv2
import pandas as p


IMAGE_SHAPE = (224, 224)   #(HEIGHT, WIDTH)

TRAINING_DATA_DIRECTORY = '/content/drive/My Drive/Colab Notebooks/FlowerClassification/data/TrainingData'

datagen_kwargs = dict(rescale=1./255, validation_split=.2)

def get_validation_generator():

    validation_datagen = ImageDataGenerator(**datagen_kwargs)

    validation_generator = validation_datagen.flow_from_directory(
        TRAINING_DATA_DIRECTORY, 
        subset='validation', 
        shuffle=True, 
        target_size=IMAGE_SHAPE
    )

    return validation_generator


def get_training_generator():
    
    training_datagen = ImageDataGenerator(**datagen_kwargs)

    training_generator = training_datagen.flow_from_directory(
        TRAINING_DATA_DIRECTORY,
        subset='training',
        shuffle=True,
        target_size=IMAGE_SHAPE
    )

    return training_generator



def get_mobile_net_model():
   
    model = keras.Sequential()
    model.add(hub.KerasLayer(
        'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4', 
        output_shape=[1280],
        trainable=False)
    )
    model.add(Dropout(0.4))
    model.add(Dense(training_generator.num_classes, activation='softmax'))

    model.build([None, 224, 224, 3])

    model.summary()

    return model



def train_model(model, training_generator=None, validation_generator=None):
    
    if (training_generator == None):
        training_generator = get_training_generator()
     
    if (validation_generator == None):
        validation_generator = get_validation_generator()

    optimizer = keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    steps_per_epoch = np.ceil(
        training_generator.samples / training_generator.batch_size
    )

    validation_steps_per_epoch = np.ceil(
        validation_generator.samples / validation_generator.batch_size
    )

    hist = model.fit(
        training_generator,
        epochs=20,
        verbose=1,
       steps_per_epoch=steps_per_epoch,
       validation_data=validation_generator,
       validation_steps=validation_steps_per_epoch
    ).history

    print('model trained')
    model.save('/content/drive/My Drive/Colab Notebooks/FlowerClassification/model_100_epochs.h5')
    print('model saved')

    #converting history.history dictionary to pandas dataframe
    hist_df = pd.DataFrame(history.history)

    # save to json
    hist_json_file = 'history_100_epochs.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    return model


def evaluate_model(model):
    final_loss, final_accuracy = model.evaluate(validation_generator, steps = validation_steps_per_epoch)
    print("Final Loss: ", final_loss)
    print("Final accuracy: ", final_accuracy * 100)


if __name__ == '__main__':
    
    model = get_mobile_net_model()
    model = train_model(model)

    evaluate_model(model)
