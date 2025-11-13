import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pandas as pd

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from matplotlib.patches import Rectangle
# from torchaudio.utils import download_asset

import requests
from io import BytesIO  # To handle raw data from HTTP requests

torch.random.manual_seed(0)


# # 1. Get the file path to an included audio example
# SAMPLE_SPEECH = librosa.example("trumpet")


# # 2. Load the audio as a waveform `y`
# #    Store the sampling rate as `sr`
# SPEECH_WAVEFORM, SAMPLE_RATE = librosa.load(SAMPLE_SPEECH)

url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

with requests.get(url, stream=True) as response:
    # torchaudio.load expects a file-like object, BytesIO can wrap raw data
    SAMPLE_SPEECH = BytesIO(response.content)
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

# SAMPLE_SPEECH = download_asset(
#     "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
# )


metadata = torchaudio.info(SAMPLE_SPEECH)
print(metadata)

# # Load audio
# SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

# Define transform
spectrogram = T.Spectrogram(n_fft=512)

# Perform transform
spec = spectrogram(SPEECH_WAVEFORM)


n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256
sample_rate = 6000

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)


mfcc = mfcc_transform(SPEECH_WAVEFORM)

# convert to dataframe
mfcc_df = pd.DataFrame(
    mfcc[0].T.numpy(), columns=[f"MFCC_{i + 1}" for i in range(mfcc.shape[1])]
)

# Each row = a time frame, each column = a feature (coefficient)
print(mfcc_df.head())
