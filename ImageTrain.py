#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:16:15 2018

@author: giovannagraciani
File (2/3)

This file trains a convolutional neural network to classify 
gender using images
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt

#image dimmensions
img_width, img_height = 150, 150

#model specs
nb_train_samples = 8500
nb_validation_samples = 1000
epochs = 12
batch_size = 32

#directories
train_data_dir='/Users/giovannagraciani/Desktop/455/train'
validation_data_dir='/Users/giovannagraciani/Desktop/455/validation'
save_dir = 'model_five.h5'

#channels_first: inputs having shape (batch, channels, height, width)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else: #channels_last
    input_shape = (img_width, img_height, 3)

#--- BUILD MODEL 2D CNN --- 
# Sequential model = linear stack of layers, each .add() creates a new layer 
# in the model
model = Sequential()
# Create 2D CNN with 32D output, with window size 4x4, that 
# takes in images of size input_shape
model.add(Conv2D(32, (4, 4), input_shape=input_shape))
model.add(Activation('relu'))
#Halve input in both dimmensions 
model.add(MaxPooling2D(pool_size=(2, 2)))
#reduce overfitting by setting 0.25x input units to 0
model.add(Dropout(0.25))
model.add(Conv2D(32, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(64, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#sigmoid activation lends itself well to binary classification tasks
model.add(Activation('sigmoid'))

#--- COMPILE MODEL --- 
# Compile binary classification model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Apply random augmentations on images during training 
# so as to reduce the change of overfitting (classifier will
# not see the same image twice)
train_datagen = ImageDataGenerator(
    #RGB scale 0-255, too high for model to process so scale 
    #by 1/255 to get values between 0 and 1
    rescale=1. / 255,
    #Shear mapping 
    shear_range=0.2,
    #randomly zoom in on image
    zoom_range=0.2,
    #flip half of images horizontally
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate augmented image batches from training directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Generate augmented image batches from validation directory
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#--- EXECUTE MODEL ---
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#--- SAVE RESULTS ---
model.save(save_dir, overwrite=True)

#--- PRINT ACCURACY ---

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

