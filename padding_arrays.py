import os
import numpy as np

numpy_directory = 'audio_data/numpy_arrays'
target_length = 305946  # Target 
min_length = 100000  # Minimum 

#iterate through each .npy file
for filename in os.listdir(numpy_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(numpy_directory, filename)
        array = np.load(filepath)

        #check if shorter than minimum length
        if len(array) < min_length:
            #delete file
            os.remove(filepath)
            print(f"Deleted file: {filename} (Length: {len(array)})")
            continue

        #check if padding needed
        if len(array) < target_length:
            #calculate no. of 0s needed to pad
            padding_length = target_length - len(array)
            # Pad the array
            padded_array = np.pad(array, (0, padding_length), mode='constant', constant_values=0)
            #save padded array
            np.save(filepath, padded_array)

print("Processing complete.")
