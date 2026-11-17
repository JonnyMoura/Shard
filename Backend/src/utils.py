import os
import pandas as pd

def load_audio_catalog(csv_path):
    """Load the audio catalog with analysis"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Audio catalog not found. Run analyze_samples.py first.")
    return pd.read_csv(csv_path)
