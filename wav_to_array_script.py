import os
import wave
import numpy as np


wav_directory = 'audio_data/raw'
numpy_directory = 'audio_data/numpy_arrays'


#for each WAV file
for filename in os.listdir(wav_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(wav_directory, filename)

        #read 
        with wave.open(filepath, 'r') as wav_file:
            samples = wav_file.getnframes()
            audio = wav_file.readframes(samples)
            sample_width = wav_file.getsampwidth()

        #set dtype
        if sample_width == 2:  # 2 bytes => int16
            dtype = np.int16
        elif sample_width == 4:  # 4 bytes => int32
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")

        #to NumPy array and normalise
        audio_as_np = np.frombuffer(audio, dtype=dtype)
        max_value = float(2 ** (8 * sample_width - 1))
        audio_normalised = audio_as_np / max_value

        #save
        npy_filename = os.path.splitext(filename)[0] + '.npy'
        npy_filepath = os.path.join(numpy_directory, npy_filename)
        np.save(npy_filepath, audio_normalised)

print("Conversion complete.")




