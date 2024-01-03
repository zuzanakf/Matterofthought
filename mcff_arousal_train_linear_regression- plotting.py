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
raw_audio_directory = 'audio_data\\raw'
resampled_directory = 'audio_data\\numpy_arrays_resampled'
target_sr = 44100  #taget 


'''# Iterate through each .npy file to find corresponding .wav files as not using some wavs due to large sr differences
for npy_file in os.listdir(numpy_directory):
    if npy_file.endswith('.npy'):
        wav_filename = npy_file.replace('.npy', '.wav')
        wav_filepath = os.path.join(raw_audio_directory, wav_filename)

        if os.path.exists(wav_filepath):
            #load the original audio file with its original sampling rate
            y, orig_sr = librosa.load(wav_filepath, sr=None)

            #check if resampling needed
            if orig_sr != target_sr:
                # Resample the audio directly
                y_resampled = librosa.resample(orig_sr=orig_sr, target_sr=target_sr, y=y)
            else:
                y_resampled = y

            #save the resampled audio data as a new .npy file
            resampled_filepath = os.path.join(resampled_directory, npy_file)
            np.save(resampled_filepath, y_resampled)

print("Resampling complete.")
print(resampled_directory)
'''
def extract_mfccs(audio_array, sr=44100, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


data = []

for filename in os.listdir(resampled_directory):
    if filename.endswith('.npy'):
        filepath = os.path.join(resampled_directory, filename)
        array = np.load(filepath)
        mfccs = extract_mfccs(array)
        # Flatten MFCCs here
        mfccs_flattened = mfccs.flatten()
        data.append({'filename': filename.replace('.npy', '.wav'), 'mfccs': mfccs_flattened})

mfcc_df = pd.DataFrame(data)

#load arousal ratings DataFrame
with open('emotion_data/dataframes/arousal_ratings.pkl', 'rb') as file:
    arousal_df = pickle.load(file)

#Merge MFCC df with the arousal ratings
merged_df = pd.merge(mfcc_df, arousal_df, on='filename')

#prepare data
X = np.vstack(merged_df['mfccs'].values)
y = merged_df['rating'].values

#training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#linear regression object
regr = linear_model.LinearRegression()

#Train
regr.fit(X_train, y_train)

#predictions
y_pred = regr.predict(X_test)

#coefficients, mean squared error, and coefficient of determination
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Create a DataFrame for easier sorting
plot_df = pd.DataFrame({
    'First MFCC Feature': X_test[:, 0],
    'Actual Arousal Ratings': y_test,
    'Predicted Arousal Ratings': y_pred
})



import matplotlib.pyplot as plt
import pandas as pd

# Number features
n_mfccs = 13  

for i in range(n_mfccs):
    #df 
    plot_df = pd.DataFrame({
        f'MFCC Feature {i+1}': X_test[:, i],
        'Actual Arousal Ratings': y_test,
        'Predicted Arousal Ratings': y_pred
    })

    # Sort the DataFrame
    plot_df = plot_df.sort_values(by=f'MFCC Feature {i+1}')

    # Plotting
    plt.figure(figsize=(8, 6))  
    plt.scatter(plot_df[f'MFCC Feature {i+1}'], plot_df['Actual Arousal Ratings'], color="black", label='Actual Arousal Ratings')
    plt.plot(plot_df[f'MFCC Feature {i+1}'], plot_df['Predicted Arousal Ratings'], color="blue", linewidth=3, label='Predicted Arousal Ratings')

    # Adding title, labels, and legend
    plt.title(f"Linear Regression Model: Predicted Arousal Ratings vs. MFCC Feature {i+1}")
    plt.xlabel(f"MFCC Feature {i+1} Value")
    plt.ylabel("Arousal Rating")
    plt.legend()
    
    plt.savefig(f"results/final_results/plots_arousal/arousal_mfcc{i+1}.png")
    # Display the plot
    plt.show()



