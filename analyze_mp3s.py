"""
analyze_mp3s.py

Appends audio features to an existing MSD-style CSV.

Expected CSV columns:

track_id
title
artist_name
tempo
loudness
duration
timbre_mean_0 ... timbre_mean_11

Usage:
    python analyze_mp3s.py /path/to/mp3/folder existing.csv
"""

import os
import sys
from pathlib import Path
import librosa
import pyloudnorm as pyln
import pandas as pd
import numpy as np
from tqdm import tqdm

COLUMNS = [
    "track_id",
    "title",
    "artist_name",
    "tempo",
    "loudness",
    "duration",
] + [f"timbre_mean_{i}" for i in range(12)]


def analyze_file(filepath):
    y, sr = librosa.load(str(filepath), sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0])

    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = float("nan")

    # Compute MFCCs as proxy for timbre (standard practice)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    timbre_means = np.mean(mfcc, axis=1)

    row = {
        "track_id": filepath.stem,   # using filename (without .mp3) as ID
        "title": filepath.stem,
        "artist_name": "unknown",
        "tempo": bpm,
        "loudness": lufs,
        "duration": duration,
    }

    for i in range(12):
        row[f"timbre_mean_{i}"] = float(timbre_means[i])

    return row


def analyze_folder(folder_path, out_csv_path):
    folder = Path(folder_path)
    mp3_files = list(folder.glob("*.mp3"))

    if not mp3_files:
        print("No MP3 files found.")
        return

    rows = []

    for f in tqdm(mp3_files, desc="Analyzing MP3s"):
        try:
            rows.append(analyze_file(f))
        except Exception as e:
            print(f"ERROR processing {f.name}: {e}")

    df = pd.DataFrame(rows, columns=COLUMNS)

    file_exists = os.path.exists(out_csv_path)

    df.to_csv(
        out_csv_path,
        mode='a',
        header=not file_exists,
        index=False
    )

    print(f"Appended {len(df)} rows to {out_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_mp3s.py /path/to/mp3/folder existing.csv")
        sys.exit(1)

    analyze_folder(sys.argv[1], sys.argv[2])
