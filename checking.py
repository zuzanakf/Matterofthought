import os
import librosa

wav_directory = 'audio_data/raw'

sampling_rates = {}

for filename in os.listdir(wav_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(wav_directory, filename)
        _, sampling_rate = librosa.load(filepath, sr=None)
        sampling_rates[filename] = sampling_rate

for filename, rate in sampling_rates.items():
    print(f"{filename}: {rate} Hz")
