import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import librosa

numpy_directory = 'audio_data\\numpy_arrays'
resampled_directory = 'audio_data\\numpy_arrays_resampled'
target_sr = 44100  #taget 


# Resample each audio file
for filename in os.listdir(numpy_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(numpy_directory, filename)
        array = np.load(filepath)

        #resample audio to target sampling rate
        resampled_array = librosa.resample(array, orig_sr=librosa.get_samplerate(filepath), target_sr=target_sr)

        #save
        resampled_path = os.path.join(resampled_directory, filename)
        np.save(resampled_path, resampled_array)

print("Resampling complete.")

def extract_mfccs(audio_array, sr=44100, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Prepare DataFrame for MFCCs
data = []

for filename in os.listdir(resampled_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(resampled_directory, filename)
        array = np.load(filepath)

        mfccs = extract_mfccs(array)
        data.append({'filename': filename, 'mfccs': mfccs})

mfcc_df = pd.DataFrame(data)
#ensuring filename is the same
mfcc_df['filename'] = mfcc_df['filename'].apply(lambda x: x.replace('.npy', '.wav'))

# Load arousal ratings DataFrame
with open('emotion_data/dataframes/arousal_ratings.pkl', 'rb') as file:
    arousal_df = pickle.load(file)

#checks
print("Audio DataFrame Size:", mfcc_df.shape)
print(mfcc_df.head())
print("Arousal Ratings DataFrame Size:", arousal_df.shape)
print(arousal_df.head())

# Merge the MFCC DataFrame with the arousal ratings
merged_df = pd.merge(mfcc_df, arousal_df, on='filename')

#checks
print("Merged DataFrame Size:", merged_df.shape)
print(merged_df.head())

# Prepare data for regression
X = np.vstack(merged_df['mfccs'].values)
y = merged_df['rating'].values

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#linear regression object
regr = linear_model.LinearRegression()

#train using the training sets
regr.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = regr.predict(X_test)


#coefficients
print("Coefficients: \n", regr.coef_)
#mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

#Plot outputs (adjusting for the fact that X_test is multidimensional)
plt.scatter(X_test[:, 0], y_test, color="black")#Plotting only the first feature of X_test for visualization
plt.plot(X_test[:, 0], y_pred, color="blue", linewidth=3)

plt.show()


