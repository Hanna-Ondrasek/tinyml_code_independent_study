import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers

batch_size = 8
# epochs = 10

x = []
y = []

max_frames = 100
num_classes = 2
n_mfcc = 13


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [Recognition]")
    plt.legend()
    plt.grid(True)


def plot_mfcc(audio_path):
    audio, sr = librosa.load(audio_path)
    audio = librosa.util.normalize(audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    shape_mfccs = mfccs.shape[1]
    if shape_mfccs > max_frames:
        mfccs = mfccs[:, :max_frames]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - shape_mfccs)))
    # sns.heatmap(mfccs, cmap="Blues")
    # plt.show()
    print(mfccs)
    x.append(mfccs)

    if "boy" in audio_path:
        y.append(0)
    elif "girl" in audio_path:
        y.append(1)


base_dir = r"c:\Users\hanna\Downloads\10_29\training"

for file in os.listdir(base_dir):
    audio_file_path = os.path.join(base_dir, file)
    plot_mfcc(audio_file_path)

x = np.array(x)
y = np.array(y)

print(y)
print(x)

print("number of dimensions:", x.ndim)
print("shape of each dimension:", x.shape)

"""
number of dimensions: 3
shape of each dimension: (35, 13, 100)
35 - number of audio clips, 13 - coefficients, 100 - number of frames 




35 layers, with each layer containing a (13 x 100) grid of numbers
"""

x = x.transpose(0, 2, 1)  # swap frames and coefs for input
print("number of dimensions after transpose:", x.ndim)


# Generating random data
# z = np.random.random((100, 10, 5))
# p = np.random.randint(2, size=(100, 1))

# x = keras.ops.convert_to_tensor(x)
# y = keras.ops.convert_to_tensor(y)


# RNN is a type of sequential model

model = keras.Sequential()
"""
Bidirectional RNN processes sequence from start to end, but also backwards, so it also gets the context around each coefficient,
in simpler terms, if we were using text, it would get the context around the full text rather than just the context after (as most RNNs are
forward feeding). This is good for our data since MFCCs are not really proper time series data (similarly to text)
The input of lstm layer has a shape of (num_timesteps, num_features) 
Time Steps. One time step is one point of observation in the sample.
Features. One feature is one observation at a time step.
https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
- i suppose the timestep would be 100 (for frames), the features would be 13 (13 coeffs per mfcc frame is an observation)
"""
model.add(
    layers.Bidirectional(
        layers.LSTM(4, return_sequences=True),
        input_shape=(
            100,
            13,
        ),  # try 2 hidden units since we dont have a large number of features,
        # some sources say to use the sqrt to the number of input features
    )  # bidirectional RNN
)
model.add(
    layers.Bidirectional(layers.LSTM(32))
)  # layer outputs one vector per entire sequence - compresses whole sequence 32 is a typical smaller value
model.add(layers.Dense(1, activation="sigmoid"))  # binary classification

model.summary()

# model = Sequential(
#     [
#         LSTM(
#             16, activation="tanh", return_sequences=True, input_shape=(13, 100)
#         ),  # first LSTM layer
#         LSTM(5, activation="tanh"),  # second LSTM layer - uses output from above
#         Dense(1, activation="sigmoid"),  #  binary classification
#     ]
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42, stratify=y
# )

# Split while still NumPy
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# convert from np.array to tensors(just to be safe)
X_train = keras.ops.convert_to_tensor(X_train)
X_test = keras.ops.convert_to_tensor(X_test)
y_train = keras.ops.convert_to_tensor(y_train)
y_test = keras.ops.convert_to_tensor(y_test)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    X_train,
    y_train,
    epochs=5,  # keep epochs low to prevent overfitting
    batch_size=16,
    validation_data=(X_test, y_test),
)

# test

base_dir = r"c:\Users\hanna\Downloads\10_29"
audio_file_path = os.path.join(base_dir, "boy_count_test.wav")
audio, sr = librosa.load(audio_file_path)
audio = librosa.util.normalize(audio)

mfccs_test = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

shape_mfccs = mfccs_test.shape[1]
if shape_mfccs > max_frames:
    mfccs_test = mfccs_test[:, :max_frames]
else:
    mfccs_test = np.pad(mfccs_test, ((0, 0), (0, max_frames - shape_mfccs)))

# expand dimensions otherwise Invalid input shape for input Tensor("data:0", shape=(13, 100), dtype=float32). Expected shape (None, 13, 100), but input has incompatible shape (13, 100)
#
mfccs_test = np.expand_dims(mfccs_test, axis=0)
print("shape of each dimension:", mfccs_test.shape)

mfccs_test = mfccs_test.transpose(0, 2, 1)

predictions = model.predict(mfccs_test)
print(predictions)
