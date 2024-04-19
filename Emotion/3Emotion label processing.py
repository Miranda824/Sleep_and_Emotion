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


def process_Z(y_arousal, y_valence, idx_arousalmin, idx_arousalmax, idx_valencemin, idx_valencemax):
    condition_arousal = np.logical_and(y_arousal >= idx_arousalmin, y_arousal < idx_arousalmax)
    condition_valence = np.logical_and(y_valence >= idx_valencemin, y_valence < idx_valencemax)
    y_arousal = np.where(np.logical_and(condition_arousal, condition_valence), 1, 0)
    return y_arousal

with open('data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain)

with open('label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL)
print(Y.shape)
X_normalized = normalize(X, norm='max', axis=0)  # Normalize along axis 0 (features)

pre_Z= process_Z(Y[:,[0]],Y[:,[1]],5,9,5,7) # arouse、valuance
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Training set/Label"
file_path = f"{folder_path}/label_training_W_N1.npy"
np.save(file_path, pre_Z)

print(pre_Z.shape)
print(np.max(pre_Z))
print(np.min(pre_Z))
Z = np.ravel(pre_Z)  # 扁平化

# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical

y_train = to_categorical(Z)
x_train = np.array(X[:])


def process_L(y_arousal, y_valence, idx_arousalmin, idx_arousalmax, idx_valencemin, idx_valencemax):
    condition_arousal = np.logical_and(y_arousal >= idx_arousalmin, y_arousal < idx_arousalmax)
    condition_valence = np.logical_and(y_valence >= idx_valencemin, y_valence < idx_valencemax)
    y_arousal = np.where(np.logical_and(condition_arousal, condition_valence), 1, 0)
    return y_arousal


with open('data_testing.npy', 'rb') as fileTrain:
    M = np.load(fileTrain)

with open('label_testing.npy', 'rb') as fileTrainL:
    N = np.load(fileTrainL)

M_normalized = normalize(M, norm='max', axis=0)  # Normalize along axis 0 (features)

pre_L = process_L(N[:, [0]],N[:, [1]],5,9,5,7)
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Testing set/Label"
file_path = f"{folder_path}/label_testing_W_N1.npy"
np.save(file_path, pre_L)

print(pre_L.shape)
print(pre_L)
print(np.max(pre_L))
print(np.min(pre_L))
L = np.ravel(pre_L)  # arousa标签

x_test = np.array(M[:])

# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
y_test = to_categorical(L)
y_test

# 验证集
with open('label_validation.npy', 'rb') as fileTrainL:
    V = np.load(fileTrainL)
print(V.shape)
pre_V= process_Z(V[:,[0]],V[:,[1]],5,9,5,7) # arouse、valuance
folder_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Validation set/Label"
file_path = f"{folder_path}/label_validation_W_N1.npy"
np.save(file_path, pre_V)

print(pre_V.shape)
print(np.max(pre_V))
print(np.min(pre_V))

# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical

y_train = to_categorical(Z)
y_train

y_train.shape

x_train = np.array(X[:])


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
model.save('emotion_W_N1.h5')