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

batch_size = 8
epochs = 10

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

# global normalization
mean = np.mean(x)
std = np.std(x)
x = (x - mean) / std

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("train:", y_train)
print("test:", y_test)

# reshape for LSTM: (samples, timesteps, features)
X_train = X_train.transpose(0, 2, 1)
X_test = X_test.transpose(0, 2, 1)

model = Sequential()
# LSTM neural network is a recurrent neural network (RNN).
model.add(LSTM(64, input_shape=(max_frames, n_mfcc)))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
    validation_data=(X_test, y_test),
)

score = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
plot_loss(history)

base_dir = r"c:\Users\hanna\Downloads\10_29"
audio_file_path = os.path.join(base_dir, "boy_count_test.wav")
audio, sr = librosa.load(audio_file_path)
audio = librosa.util.normalize(audio)

mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

shape_mfccs = mfccs.shape[1]
if shape_mfccs > max_frames:
    mfccs = mfccs[:, :max_frames]
else:
    mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - shape_mfccs)))

mfccs = (mfccs - mean) / std

mfccs = mfccs.T.reshape(1, max_frames, n_mfcc)

probs = model.predict(mfccs)[0][0]
predicted_label = 1 if probs >= 0.5 else 0

print("Predicted label: ", predicted_label)
print("Probability:", probs)


y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
