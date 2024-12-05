import numpy as np
import os
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Dense, Flatten, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_label(y_valence, y_arousal, idx_valencemin1, idx_valencemax1, idx_arousalmin1, idx_arousalmax1,
                                        idx_valencemin2, idx_valencemax2, idx_arousalmin2, idx_arousalmax2,
                                        idx_valencemin3, idx_valencemax3, idx_arousalmin3, idx_arousalmax3,
                                        idx_valencemin4, idx_valencemax4, idx_arousalmin4, idx_arousalmax4):
    condition_valence1 = np.logical_and(y_valence >= idx_valencemin1, y_valence < idx_valencemax1)
    condition_arousal1 = np.logical_and(y_arousal >= idx_arousalmin1, y_arousal < idx_arousalmax1)

    condition_valence2 = np.logical_and(y_valence >= idx_valencemin2, y_valence < idx_valencemax2)
    condition_arousal2 = np.logical_and(y_arousal >= idx_arousalmin2, y_arousal < idx_arousalmax2)

    condition_valence3 = np.logical_and(y_valence >= idx_valencemin3, y_valence < idx_valencemax3)
    condition_arousal3 = np.logical_and(y_arousal >= idx_arousalmin3, y_arousal < idx_arousalmax3)

    condition_valence4 = np.logical_and(y_valence >= idx_valencemin4, y_valence < idx_valencemax4)
    condition_arousal4 = np.logical_and(y_arousal >= idx_arousalmin4, y_arousal < idx_arousalmax4)

    # Create a condition that satisfies any of the three classes
    any_condition = (np.any(np.logical_or(condition_arousal1, condition_valence1)) or np.any(np.logical_or(condition_arousal2, condition_valence2))
                     or np.any(np.logical_or(condition_arousal3, condition_valence3)) or np.any(np.logical_or(condition_arousal4, condition_valence4)))

    # Create an array with default class as 0
    y_new = np.zeros_like(y_arousal)

    # Assign labels based on the condition
    y_new[any_condition] = 1

    return y_new

# Load training data
with open('data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain)
print(X.shape)
with open('label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL)
print(Y.shape)

pre_Y = process_label(Y[:, [0]], Y[:, [1]], 1, 3, 7, 9,
                                            1, 3, 3, 5,
                                            3, 5, 1, 5,
                                            7, 9, 5, 7)
Y_flattened = np.ravel(pre_Y)  # Flatten the processed labels

x_train = np.array(X[:])
y_train = to_categorical(Y_flattened)

# Load testing data
with open('data_testing.npy', 'rb') as fileTest:
    M = np.load(fileTest)
with open('label_testing.npy', 'rb') as fileTestL:
    N = np.load(fileTestL)

pre_N = process_label(N[:, [0]], N[:, [1]], 1, 3, 7, 9,
                                            1, 3, 3, 5,
                                            3, 5, 1, 5,
                                            7, 9, 5, 7)
N_flattened = np.ravel(pre_N)

x_test = np.array(M[:])
y_test = to_categorical(N_flattened)

# Load validation data
with open('data_validation.npy', 'rb') as fileVal:
    O = np.load(fileVal)

with open('label_validation.npy', 'rb') as fileValL:
    V = np.load(fileValL)

pre_V = process_label(V[:, [0]], V[:, [1]], 1, 3, 7, 9,
                                            1, 3, 3, 5,
                                            3, 5, 1, 5,
                                            7, 9, 5, 7)
V_flattened = np.ravel(pre_V)

x_val = np.array(O[:])
y_val = to_categorical(V_flattened)

# Standardize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # Use scaler fitted on training data
x_val = scaler.transform(x_val)   # Use scaler fitted on training data

# Reshape data to 3D for Conv1D
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# Create the model
batch_size = 256
num_classes = 2
epochs = 30
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

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))


# Ensure the directory exists
save_dir = './emotion_model'
os.makedirs(save_dir, exist_ok=True)
# Save the model
model.save(os.path.join(save_dir, 'emotion_N1_N3.h5'))
