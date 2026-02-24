import os
import h5py
import csv
import numpy as np

# Configuration
msd_subset_path = "C:/Users/scott/OneDrive/Desktop/Music Data/MillionSongSubset"
output_file = 'msd_flattened.csv'

# Define the features we want to extract
# We'll take metadata + numerical features for distance math
header = ['track_id', 'title', 'artist_name', 'tempo', 'loudness', 'duration']
# Timbre has 12 dimensions; we'll add them as separate columns
header += [f'timbre_mean_{i}' for i in range(12)]

def extract_song_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # 1. Try to find the track_id (it moves around in some versions)
        try:
            track_id = f['analysis']['songs']['track_id'][0].decode('utf-8')
        except (KeyError, ValueError):
            try:
                track_id = f['metadata']['songs']['track_id'][0].decode('utf-8')
            except:
                track_id = "Unknown"

        # 2. Get standard metadata
        # We use .get() or a try-block to prevent the script from crashing
        try:
            title = f['metadata']['songs']['title'][0].decode('utf-8')
            artist = f['metadata']['songs']['artist_name'][0].decode('utf-8')
        except:
            title, artist = "Unknown", "Unknown"
        
        # 3. Get numerical features for Euclidean Distance
        # These are almost always in analysis/songs
        analysis_songs = f['analysis']['songs']
        tempo = analysis_songs['tempo'][0]
        loudness = analysis_songs['loudness'][0]
        duration = analysis_songs['duration'][0]
        
        # 4. Get Timbre (Texture) data
        timbre_data = f['analysis']['segments_timbre'][:]
        if timbre_data.size > 0:
            avg_timbre = np.mean(timbre_data, axis=0).tolist()
        else:
            avg_timbre = [0] * 12 
            
        return [track_id, title, artist, tempo, loudness, duration] + avg_timbre
# Main processing loop
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
    print("Starting extraction...")
    count = 0
    for root, dirs, files in os.walk(msd_subset_path):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                try:
                    song_row = extract_song_data(file_path)
                    writer.writerow(song_row)
                    count += 1
                    if count % 500 == 0:
                        print(f"Processed {count} songs...")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

print(f"Finished! Saved {count} songs to {output_file}")