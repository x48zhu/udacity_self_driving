import csv
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Lambda, Dropout, Activation

model = Sequential()
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#
#
# model = Sequential()
# input_shape=(160,320,3)
# nb_filters1 = 16
# nb_filters2 = 8
# nb_filters3 = 4
# nb_filters4 = 2
# pool_size = (2, 2)
# kernel_size = (3, 3)
# nb_classes = 1
#
# # Starting with the convolutional layer
# # The first layer will turn 1 channel into 16 channels
# model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape))
# # Applying ReLU
# model.add(Activation('relu'))
# # The second conv layer will convert 16 channels into 8 channels
# model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
# # Applying ReLU
# model.add(Activation('relu'))
# # The second conv layer will convert 8 channels into 4 channels
# model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
# # Applying ReLU
# model.add(Activation('relu'))
# # The second conv layer will convert 4 channels into 2 channels
# model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
# # Applying ReLU
# model.add(Activation('relu'))
# # Apply Max Pooling for each 2 x 2 pixels
# model.add(MaxPooling2D(pool_size=pool_size))
# # Apply dropout of 25%
# model.add(Dropout(0.25))
#
# # Flatten the matrix. The input has size of 360
# model.add(Flatten())
# # Input 360 Output 16
# model.add(Dense(16))
# # Applying ReLU
# model.add(Activation('relu'))
# # Input 16 Output 16
# model.add(Dense(16))
# # Applying ReLU
# model.add(Activation('relu'))
# # Input 16 Output 16
# model.add(Dense(16))
# # Applying ReLU
# model.add(Activation('relu'))
# # Apply dropout of 50%
# model.add(Dropout(0.5))
# # Input 16 Output 1
# model.add(Dense(nb_classes))
#
#
# model.summary()
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
