import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. Load your processed data
df = pd.read_csv('msd_flattened.csv')

# 2. Select only the numerical columns for math
# We include tempo, loudness, duration, and the 12 timbre dimensions
feature_cols = ['tempo', 'loudness', 'duration'] + [f'timbre_mean_{i}' for i in range(12)]
features = df[feature_cols]

# 3. CRITICAL: Standardization
# This ensures that 'tempo' (large numbers) doesn't drown out 'loudness' (small numbers)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. Build the "Map" (The Model)
# We use the 'euclidean' metric to find neighbors
model = NearestNeighbors(n_neighbors=6, metric='euclidean')
model.fit(scaled_features)

def find_similar_songs(song_title):
    # Find the index of the song you want to look up
    try:
        song_idx = df[df['title'].str.lower() == song_title.lower()].index[0]
    except IndexError:
        return "Song not found in dataset."

    # Search the map for the 5 closest points
    distances, indices = model.kneighbors([scaled_features[song_idx]])

    print(f"\nSongs similar to '{df.iloc[song_idx]['title']}' by {df.iloc[song_idx]['artist_name']}:")
    print("-" * 50)
    
    # Skip the first result because it will be the song itself (distance = 0)
    for i in range(1, len(distances[0])):
        neighbor_idx = indices[0][i]
        dist = distances[0][i]
        neighbor_song = df.iloc[neighbor_idx]
        print(f"{i}. {neighbor_song['title']} by {neighbor_song['artist_name']} (Distance: {dist:.2f})")

# TEST IT!
# Pick a song title that you saw in your CSV
find_similar_songs("Wake Me Up When September Ends (Live at Foxboro_ MA 9/3/05)")