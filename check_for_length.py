import os
import wave
import numpy as np

numpy_directory = 'audio_data/numpy_arrays'
wav_directory = 'audio_data/raw'

# Dictionary group files by characteristics
file_groups = {}

#process
for filename in os.listdir(wav_directory):
    if filename.endswith(".wav"):
        wav_path = os.path.join(wav_directory, filename)
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(numpy_directory, npy_filename)

        #Check .npy file exists
        if os.path.exists(npy_path):
            with wave.open(wav_path, 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()

            array = np.load(npy_path)

            #key representing  characteristics
            key = (rate, channels, len(array))

            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(filename)

#print summary
for key, filenames in file_groups.items():
    rate, channels, length = key
    print(f"Group: Sampling Rate {rate} Hz, Channels {channels}, Array Length {length}")
    print(f"  Number of files: {len(filenames)}")
    print(f"  Example files: {filenames[:5]}")  # Print first 5 filenames as examples
    print("-" * 30)
