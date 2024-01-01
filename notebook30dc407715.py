# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

image = PIL.Image.open('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_virus_1681.jpeg')
image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'
training_generator = ImageDataGenerator(rescale=1/255)
data_train = training_generator.flow_from_directory(training_dir, target_size=(120,120), batch_size=8, class_mode='binary')

validation_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'
validation_generator = ImageDataGenerator(rescale=1/255)
data_valid = validation_generator.flow_from_directory(validation_dir, target_size=(120,120), batch_size=8, class_mode='binary')

testing_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'
testing_generator = ImageDataGenerator(rescale=1/255)
data_test = testing_generator.flow_from_directory(testing_dir, target_size=(120,120), batch_size=8, class_mode='binary')

#building CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(120,120,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
                            ])

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(data_train, epochs=2, validation_data=data_valid)

model.evaluate(data_test)

predictions = model.predict(data_test)

predictions

x = data_test.next()
for i in range(0,1):
    image = x[i]
    for j in range (0,16):
        plt.imshow(image[j])
        plt.show()
        print("Pneumonia Probability is: ", predictions[i])

