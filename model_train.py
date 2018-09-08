# Dependencies

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import time
import math

# Data loader
import preprocess

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


## Normal sEMG dataset
file_path = "./SEMG_DB1/N_TXT/"

input_array, output_array = preprocess.preprocess(file_path)
print(input_array.shape)
print(input_array)

## One Hot Encode Prediction
print(output_array)
# convert output_array to one hot encoded classes
values = array(output_array)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# Retrieve from input array shape
time_steps = input_array.shape[1]
n_features = input_array.shape[2]

# code for building an LSTM with 100 neurons and dropout.

model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(time_steps, n_features)))
model.add(Dropout(0.5))
#model.add(LSTM(100)) dramatically worse results
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(input_array, onehot_encoded, batch_size=16, epochs=20)

## Abnormal sEMG dataset
abnormal_path = "./SEMG_DB1/A_TXT/"

abnormal_input_array, abnormal_output_array = preprocess.preprocess(file_path)

# convert output_array to one hot encoded classes
values = array(output_array)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
abnormal_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(abnormal_onehot_encoded)

# test on abnormal dataset

score = model.evaluate(abnormal_input_array, abnormal_onehot_encoded, batch_size=16)
print(score)
