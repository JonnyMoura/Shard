import os
import random
import librosa
import numpy as np
import pandas as pd

def load_audio_catalog(csv_path):
    """Load the audio catalog with analysis"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Audio catalog not found. Run analyze_samples.py first.")
    return pd.read_csv(csv_path)

def get_tracks_from_solution(solution, catalog_df, tracks_per_solution=2):
    """Extract track paths from solution parameters"""
    # Last N parameters are track indices
    track_indices = solution[-tracks_per_solution:].astype(int)
    track_indices = np.clip(track_indices, 0, len(catalog_df) - 1)
    
    # Get file paths for selected tracks
    track_paths = []
    for idx in track_indices:
        track_paths.append(catalog_df.iloc[idx]['file_path'])
    
    return track_paths

def pick_category_tracks(catalog_df, category, n=2, diversity=0.2):
    """Pick tracks from a specific category with corrected logic"""
    # Filter by dominant category and create a copy to avoid warnings
    filtered_df = catalog_df[catalog_df['dominant_category'] == category].copy()
    
    if len(filtered_df) < n:
        print(f"Warning: Only {len(filtered_df)} tracks found in {category} category. Adding random tracks.")
        # Not enough tracks in this category, add some random ones
        additional = catalog_df.sample(n - len(filtered_df))
        filtered_df = pd.concat([filtered_df, additional])
    
    # Sort by the descriptors that are actually available
    if category == "low":
        # For low category, prefer tracks with lower centroid values and higher spectral flux (continuous)
        filtered_df['continuous_score'] = filtered_df['spectral_flux'] - filtered_df['centroid'] / 100
        sorted_df = filtered_df.sort_values(by="continuous_score", ascending=False)
    elif category == "mid":
        # For mid category, prefer tracks with moderate centroid and high spectral flux (continuous)
        filtered_df['centroid_distance'] = abs(filtered_df['centroid'] - 1750)
        filtered_df['continuous_score'] = filtered_df['spectral_flux'] - filtered_df['centroid_distance']
        sorted_df = filtered_df.sort_values(by="continuous_score", ascending=False)
    elif category == "high":
        # For high category, prefer tracks with higher centroid and high spectral flux (continuous)
        filtered_df['continuous_score'] = filtered_df['spectral_flux'] + filtered_df['centroid'] / 100
        sorted_df = filtered_df.sort_values(by="continuous_score", ascending=False)
    elif category == "rhythmic":
        
        filtered_df['rhythmic_score'] = filtered_df['onset_rate'] * 100 - filtered_df['spectral_flux']
        sorted_df = filtered_df.sort_values(by="rhythmic_score", ascending=False)
    else:
        # Default: random order
        sorted_df = filtered_df.sample(frac=1)
    
    # Take the top portion based on diversity parameter
    pool_size = max(n, int(len(sorted_df) * diversity))
    candidate_pool = sorted_df.iloc[:pool_size]
    
    # Select n tracks from the pool
    selected_indices = random.sample(range(len(candidate_pool)), min(n, len(candidate_pool)))
    selected_tracks = candidate_pool.iloc[selected_indices]
    
    return selected_tracks['file_path'].tolist()

def load_audio_tracks(track_paths, target_sr=None):
    """Load multiple audio tracks as numpy arrays"""
    sources = []
    sample_rates = []
    for path in track_paths:
        if not os.path.exists(path):
            print(f"WARNING: Audio file not found: {path}")
            continue
        try:
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            sources.append(audio)
            sample_rates.append(sr)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
    
    # Ensure we have at least one source
    if len(sources) == 0:
        print("WARNING: No audio sources could be loaded!")
        # Create a default source (silence)
        sr = target_sr if target_sr else 44100
        sources = [np.zeros(sr)]  # 1 second of silence
        sample_rates = [sr]
        
    return sources, sample_rates

