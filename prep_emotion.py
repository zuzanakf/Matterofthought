import pandas as pd
import pickle

arousal_file_path = 'emotion_data/raw_arousal/Arousal.csv'
valence_file_path = 'emotion_data/raw_valence/Valence.csv'

# Load into DataFrame and rename columns
arousal_df = pd.read_csv(arousal_file_path)
arousal_df.columns = ['filename', 'rating']  # Renaming columns

valence_df = pd.read_csv(valence_file_path)
valence_df.columns = ['filename', 'rating']  # Renaming columns

# Optionally, inspect the DataFrame
print(arousal_df.head())
print(valence_df.head())

# Save arousal DataFrame with pickle
pickle_file_path_a = 'emotion_data/dataframes/arousal_ratings.pkl'
with open(pickle_file_path_a, 'wb') as file:
    pickle.dump(arousal_df, file)

print("Arousal DataFrame saved as pickle file at:", pickle_file_path_a)

# Save valence DataFrame with pickle
pickle_path_v = 'emotion_data/dataframes/valence_ratings.pkl'
with open(pickle_path_v, 'wb') as file:
    pickle.dump(valence_df, file)

print("Valence DataFrame saved as pickle file at:", pickle_path_v)
