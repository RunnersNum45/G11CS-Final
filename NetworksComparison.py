# Cat Dog Classifier Network Training
# In this program I will use keras to train a neural network to identify pictures of cats and dogs. I will look at the accuracy reached and training time of different network structures.


# Setup
# I first need to import everything that is used and seed the random numbers for reproducibility. An array is also initialized to record the results of each network.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import time

# Basics
# I need to setup the general method I will be using to train. I will normalize the image size, set the epoch count, batch size, and the location of the data. The data should be structured as follows.
# ../
#     ThisFile.py
#     cats_and_dogs_filtered/
#        train/
#            cats/
#                images...
#            dogs/
#                images...
#        validation/
#            cats/
#                images...
#            dogs/
#                images...
#        vectorize.py

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = './cats_and_dogs_filtered/train'
validation_data_dir = './cats_and_dogs_filtered/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Data
# The data needs to be preprocessed and stored. I set up a preprocess for the training data that has a large amount of randomization. This is to avoid overfitting and to vary the dataset more. I both cases I rescale the pixels to contain data between 0-1 as opposed to 0-255.

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# Helpers
# This is a function to plot the accuracy of a network over time as it is trained and a class to record epoch time. I will use the both later.
def plotacc(history):
    # Plot training & validation accuracy values
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
class DataRecord(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Initialising the CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.summary()

# Compiling the CNN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
time_callback = DataRecord(name)

record = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose = 1,
    callbacks = [time_callback])

time_callback.history = record.history
time_callback.acc = record.history["acc"][-1]
print ("Training Time:", sum(time_callback.times))
print ("Accuracy:", time_callback.acc)
plotacc(record.history)