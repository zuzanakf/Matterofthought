import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import librosa

target_sr = 44100

def extract_mfccs(audio_array, sr=44100, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


def process_new_audio(file_path, target_sr=44100):
    y, orig_sr = librosa.load(file_path, sr=None)
    if orig_sr != target_sr:
        y = librosa.resample(orig_sr=orig_sr, target_sr=target_sr, y=y)

    mfccs = extract_mfccs(y)
    mfccs_flattened = mfccs.flatten()
    return mfccs_flattened

model = 'models/arousal_mfccs_model.pkl'
# Load the saved model
with open(model, 'rb') as file:
    loaded_model = pickle.load(file)


new_audio_directory = 'our_audio_data'  

# Process and predict arousal for each audio file
for audio_file in os.listdir(new_audio_directory):
    if audio_file.endswith('.wav'):
        file_path = os.path.join(new_audio_directory, audio_file)
        new_audio_features = process_new_audio(file_path)
        new_audio_features = new_audio_features.reshape(1, -1)
        predicted_arousal = loaded_model.predict(new_audio_features)
        print(f"Predicted arousal for {audio_file}: {predicted_arousal[0]}")
