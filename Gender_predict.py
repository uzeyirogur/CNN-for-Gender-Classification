from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation and Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

# Training and Test Data
training_set = train_datagen.flow_from_directory('data/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=1,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='binary')

# Train the CNN
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)

# Predictions and Evaluation
import numpy as np
import pandas as pd

test_set.reset()
pred = classifier.predict_generator(test_set, verbose=1)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0

test_labels = []

for i in range(0, int(203)):
    test_labels.extend(np.array(test_set[i][1]))

print('test_labels')
print(test_labels)

filenames = test_set.filenames
results = pd.DataFrame()
results['filenames'] = filenames
results['predictions'] = pred
results['actual'] = test_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print(cm)

#--------------------------

# Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compile the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Data Augmentation and Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

# Training and Test Data
training_set = train_datagen.flow_from_directory("data/training_set",
                                                 target_size=(64, 64),
                                                 batch_size=1,
                                                 class_mode="binary")

test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='binary')

# Train the CNN
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)

# Predictions and Evaluation
import pandas as pd
import numpy as np

test_set.reset()
pred = classifier.predict_generator(test_set, verbose=1)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0

test_labels = []

for i in range(0, int(204)):
    test_labels.extend(np.array(test_set[i][1]))

print('test_labels')
print(test_labels)

filenames = test_set.filenames
results = pd.DataFrame()
results['filenames'] = filenames
results['predictions'] = pred
results['actual'] = test_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print(cm)
