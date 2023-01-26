import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Define the model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

input_shape = model.input_shape[1:3]

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the data
data_dir = "ufo"
batch_size = 32
data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = data_gen.flow_from_directory(directory=data_dir,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           target_size=(256, 256),
                                           class_mode='categorical')

# Train the model
model.fit(train_data, epochs=3)

# Query the model
#model.predict(...)
