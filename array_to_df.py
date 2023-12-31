import os
import numpy as np
import pandas as pd
import pickle

numpy_directory = 'audio_data/numpy_arrays'
pickle_file_path = 'audio_data/dataframes/pickled_audio_dataframe.pkl' 


data = []

#for each .npy
for filename in os.listdir(numpy_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(numpy_directory, filename)
        array = np.load(filepath)

        # Create a record with filename and array
        record = {'filename': filename, 'array': array}
        data.append(record)

#create df
df = pd.DataFrame(data)

#save with pickle
with open(pickle_file_path, 'wb') as file:
    pickle.dump(df, file)

print(f"DataFrame saved as pickle at: {pickle_file_path}")

