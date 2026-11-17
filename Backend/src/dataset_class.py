import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm


def classify_audio_category(centroid, spectral_flux, onset_rate, zcr):
    if 700 < spectral_flux < 1500 and onset_rate > 8.0 and zcr > 0.1:
        return "rhythmic"
    if centroid <= 1200:
        return "low"
    if centroid >= 3000:
        return "high"
    return "mid"


def analyze_audio_file(file_path):
    """Analyze audio file using corrected classification logic with rolloff"""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
       
        n_fft = 1024
        hop_length = 512
        
        
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        
        centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85))
        spectral_flux = np.mean(np.sum(np.diff(S, axis=1)**2, axis=0)) if S.shape[1] > 1 else 0.0
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_rate = len(onset_frames) / (len(y) / sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop_length))

        
        dominant_category = classify_audio_category(
            centroid, spectral_flux, onset_rate, zcr
        )
        
        return {
            "file_path": file_path,
            "centroid": centroid,
            "rolloff": rolloff,  
            "spectral_flux": spectral_flux,
            "onset_rate": onset_rate,
            "dominant_category": dominant_category
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def analyze_folder(folder_path, output_csv):
    """Analyze all audio files in a folder and save to CSV"""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    results = []
    for file_path in tqdm(audio_files, desc="Analyzing audio files"):
        result = analyze_audio_file(file_path)
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Analysis complete. {len(results)} files analyzed and saved to {output_csv}")
    print(f"Category distribution: {df['dominant_category'].value_counts().to_dict()}")
    return df
