import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Loading dfs
with open('audio_data/dataframes/pickled_audio_dataframe.pkl', 'rb') as file:
    audio_df = pickle.load(file)


with open('emotion_data/dataframes/arousal_ratings.pkl', 'rb') as file:
    arousal_df = pickle.load(file)


#checks
print("Audio DataFrame Size:", audio_df.shape)
print(audio_df.head())
print("Arousal Ratings DataFrame Size:", arousal_df.shape)
print(arousal_df.head())

#ensuring filename is the same
audio_df['filename'] = audio_df['filename'].apply(lambda x: x.replace('.npy', '.wav'))

#merge based on filenames
merged_df = pd.merge(audio_df, arousal_df, on='filename')

print("Merged DataFrame Size:", merged_df.shape)
print(merged_df.head())

if merged_df.empty:
    raise ValueError("Merged DataFrame is empty. Check the filename columns in both DataFrames.")

#extract features from audio data (flatten array)
X = np.array([array.flatten() for array in merged_df['array']])

#arousal ratings
y = merged_df['rating'].values

#split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#linear regression object
regr = linear_model.LinearRegression()

#train using the training sets
regr.fit(X_train, y_train)

#make predictions using the testing set
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
