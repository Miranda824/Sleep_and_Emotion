import numpy as np
import pyeeg as pe
import pickle as pickle
import pandas as pd
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import tensorflow as tf
import time


def process_Z(y_arousal, y_valence, idx_arousalmin1, idx_arousalmax1, idx_valencemin1, idx_valencemax1,
              idx_arousalmin2, idx_arousalmax2, idx_valencemin2, idx_valencemax2,
              idx_arousalmin3, idx_arousalmax3, idx_valencemin3, idx_valencemax3):
    condition_arousal1 = np.logical_and(y_arousal >= idx_arousalmin1, y_arousal < idx_arousalmax1)
    condition_valence1 = np.logical_and(y_valence >= idx_valencemin1, y_valence < idx_valencemax1)

    condition_arousal2 = np.logical_and(y_arousal >= idx_arousalmin2, y_arousal < idx_arousalmax2)
    condition_valence2 = np.logical_and(y_valence >= idx_valencemin2, y_valence < idx_valencemax2)

    condition_arousal3 = np.logical_and(y_arousal >= idx_arousalmin3, y_arousal < idx_arousalmax3)
    condition_valence3 = np.logical_and(y_valence >= idx_valencemin3, y_valence < idx_valencemax3)

    # Create a condition that satisfies any of the three classes
    any_condition = np.any(np.logical_or(condition_arousal1, condition_valence1)) or np.any(np.logical_or(condition_arousal2, condition_valence2)) or np.any(np.logical_or(condition_arousal3, condition_valence3))

    # Create an array with default class as 0
    y_class = np.zeros_like(y_arousal)

    # Assign labels based on the condition
    y_class[any_condition] = 1

    return y_class


with open('data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain)

with open('label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL)
print(Y.shape)
X_normalized = normalize(X, norm='max', axis=0)  # Normalize along axis 0 (features)

pre_Z = process_Z(Y[:, [0]], Y[:, [1]], 1, 3, 3, 5,
                                        3, 5, 7, 9,
                                        7, 9, 1, 5)
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Training set/Label"
file_path = f"{folder_path}/label_training_N2_R.npy"
np.save(file_path, pre_Z)
print(pre_Z.shape)
print(np.max(pre_Z))
print(np.min(pre_Z))
Z = np.ravel(pre_Z)  # 扁平化

# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical

y_train = to_categorical(pre_Z, num_classes=2)
x_train = np.array(X[:])


def process_L(y_arousal, y_valence, idx_arousalmin1, idx_arousalmax1, idx_valencemin1, idx_valencemax1,
              idx_arousalmin2, idx_arousalmax2, idx_valencemin2, idx_valencemax2,
              idx_arousalmin3, idx_arousalmax3, idx_valencemin3, idx_valencemax3):
    condition_arousal1 = np.logical_and(y_arousal >= idx_arousalmin1, y_arousal < idx_arousalmax1)
    condition_valence1 = np.logical_and(y_valence >= idx_valencemin1, y_valence < idx_valencemax1)

    condition_arousal2 = np.logical_and(y_arousal >= idx_arousalmin2, y_arousal < idx_arousalmax2)
    condition_valence2 = np.logical_and(y_valence >= idx_valencemin2, y_valence < idx_valencemax2)

    condition_arousal3 = np.logical_and(y_arousal >= idx_arousalmin3, y_arousal < idx_arousalmax3)
    condition_valence3 = np.logical_and(y_valence >= idx_valencemin3, y_valence < idx_valencemax3)

    # Create a condition that satisfies any of the three classes
    any_condition = np.any(np.logical_or(condition_arousal1, condition_valence1)) or np.any(np.logical_or(condition_arousal2, condition_valence2)) or np.any(np.logical_or(condition_arousal3, condition_valence3))

    # Create an array with default class as 0
    y_class = np.zeros_like(y_arousal)

    # Assign labels based on the condition
    y_class[any_condition] = 1

    return y_class


with open('data_testing.npy', 'rb') as fileTrain:
    M = np.load(fileTrain)

with open('label_testing.npy', 'rb') as fileTrainL:
    N = np.load(fileTrainL)


M_normalized = normalize(M, norm='max', axis=0)  # Normalize along axis 0 (features)

pre_L = process_L(N[:, [0]], N[:, [1]], 1, 3, 3, 5,
                                        3, 5, 7, 9,
                                        7, 9, 1, 5)
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Testing set/Label"
file_path = f"{folder_path}/label_testing_N2_R.npy"
np.save(file_path, pre_L)
print(pre_L.shape)
print(np.max(pre_L))
print(np.min(pre_L))
L = np.ravel(pre_L)  # arousa标签

Arousal_Test = np.ravel(N[:, [0]])
Valence_Test = np.ravel(N[:, [1]])
Domain_Test = np.ravel(N[:, [2]])
Like_Test = np.ravel(N[:, [3]])

x_test = np.array(M[:])


# 验证集
with open('label_validation.npy', 'rb') as fileTrainL:
    V = np.load(fileTrainL)
print(V.shape)
pre_V = process_L(V[:, [0]], V[:, [1]], 1, 3, 3, 5,
                                        3, 5, 7, 9,
                                        7, 9, 1, 5)
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Validation set/Label"
file_path = f"{folder_path}/label_validation_N2_R.npy"
np.save(file_path, pre_V)
print(pre_V.shape)
print(np.max(pre_V))
print(np.min(pre_V))

# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
y_test = to_categorical(L, num_classes=2)
y_test

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

x_train.shape

batch_size = 256
num_classes = 2
epochs = 30
input_shape=(x_train.shape[1], 1)


from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Flatten, Dropout


input_shape = (x_train.shape[1], x_train.shape[2])

model = Sequential()
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu', strides=2))
model.add(BatchNormalization())
model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(512, kernel_size=3, padding='same', activation='relu', strides=2))
model.add(BatchNormalization())
model.add(Conv1D(512, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()



import tensorflow as tf

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer='adam',
#               metrics=['accuracy'])
print(x_train.shape)
print(y_train.shape)
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,validation_data=(x_test,y_test))

 #Load the library
from tensorflow.keras.models import load_model
# Save the model using TensorFlow SavedModel format
model.save('emotion_N1_N2.h5')