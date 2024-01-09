import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# upload the data directly from tensorflow (no need of csv files)
mnist = tf.keras.datasets.mnist

# split the data in training data and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x represents the digit and y the label

# normalize data pixels and scale the digits between 0 and 9
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape the data for CNN (add a channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# create the model
model = tf.keras.models.Sequential()
# add layers
# convolutional layers
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# flatten before the dense layers
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# dense layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # this will be the output layer
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train and save the model
model.fit(x_train, y_train, batch_size=64, epochs=7)
model.save('handwritten_model_cnn')





