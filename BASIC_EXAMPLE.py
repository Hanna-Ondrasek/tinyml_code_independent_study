import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# Define the batch size
batch_size = 8

# Define the number of epochs
epochs = 10


# features and labels
x = []
y = []

# for padding
max_frames = 20

# define the number of classes
num_classes = 2


# for later :3
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [Recognition]")
    plt.legend()
    plt.grid(True)


def plot_mfcc(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # type: class 'numpy.ndarray'
    # append the features and labels to the x and y lists

    # padding
    # before padding: The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (5, 13) + inhomogeneous part.
    shape_mfccs = mfccs.shape[1]
    if shape_mfccs > max_frames:
        mfccs = mfccs[:, :max_frames]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - shape_mfccs)))
    # pad_mfcc(mfccs)

    x.append(mfccs)
    # oh my god just checking to see if in the training file string if
    # the filename contains a label e.g. male or female voice
    # is actually genius
    if "boy" in audio_path:
        y.append(0)  # 0 for "boy"
    elif "girl" in audio_path:
        y.append(1)  # 1 for "girl"


labels = ["boy", "girl"]

# print(type(mfccs))
# # Shape: (13, 67)
# #  the optimum number of MFCCs is well known to be 13
# # meaning: 67 elements for each of the 13 1D arrays (13 rows, 67 cols)
# # frame = short audio slices of 20 ms
# print("Shape:", mfccs.shape)
# # numpy array to pd dataframe
# # Using pd.DataFrame()
# df1 = pd.DataFrame(mfccs)

# # Display the DataFrame
# print(df1)

# # Plot MFCCs
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis="time", cmap="viridis")
# plt.colorbar(format="%+2.0f dB")
# plt.title("MFCC")
# plt.xlabel("Time")
# plt.ylabel("MFCC Coefficient")
# plt.show()


# Example usage
# base_dir = r"c:\Users\hanna\Downloads\10_29"
# audio_file_path = os.path.join(base_dir, "girl_count1.wav")
# plot_mfcc(audio_file_path)


base_dir = r"c:\Users\hanna\Downloads\10_29\training"

for file in os.listdir(base_dir):
    audio_file_path = os.path.join(base_dir, file)
    plot_mfcc(audio_file_path)

list_size = len(y)
print(list_size)
print(y)

# Convert the lists to numpy arrays
x = np.array(x)
y = np.array(y)

print("x dimensions:", x.ndim)


# https://www.kaggle.com/code/romeodavid/keras-librosa-mfcc-feature-extraction

# time to actually train the model
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, random_state=42
)

# Reshape the data to fit the input shape of the CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


model = Sequential()
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2], 1),
    )
)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
    validation_data=(X_test, y_test),
)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
plot_loss(history)

# silly goofy territory
base_dir = r"c:\Users\hanna\Downloads\10_29"
audio_file_path = os.path.join(base_dir, "boy_count_test.wav")
audio, sr = librosa.load(audio_file_path)

# mfcc stuff
# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
# type: class 'numpy.ndarray'
# append the features and labels to the x and y lists

# padding
# before padding: The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (5, 13) + inhomogeneous part.
shape_mfccs = mfccs.shape[1]
if shape_mfccs > max_frames:
    mfccs = mfccs[:, :max_frames]
else:
    mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - shape_mfccs)))
    # pad_mfcc(mfccs)


# Normalize MFCC features
# Calculate mean and standard deviation of MFCC features
mfccs = np.array(mfccs)
mean = np.mean(mfccs)
std = np.std(mfccs)
mfccs_norm = (mfccs - mean) / std

# Reshape MFCC features for model input
mfccs_norm = mfccs_norm.reshape(1, mfccs_norm.shape[0], mfccs_norm.shape[1], 1)


# Get predicted class probabilities
probs = model.predict(mfccs_norm)
predicted_label = np.argmax(probs)

print("Predicted label: ", predicted_label)


# plot_mfcc(audio_file_path)


# waveform, sample_rate = torchaudio.load("crumple.wav", normalize=True)
# transform = T.MFCC(
#     sample_rate=sample_rate,
#     n_mfcc=13,
#     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
# )
# mfcc = transform(waveform)
