import os
import numpy as np
import soundfile as sf
import random
from pedalboard import Gain,Reverb, HighpassFilter, LowpassFilter, LadderFilter, Compressor,Chorus,Distortion
import librosa

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def grain_envelope(size,shape):
    if shape == 'smooth':
        
        x = np.linspace(-3, 3, size)
        env = np.exp(-x**2)
        return env / np.max(env)
    elif shape == 'hann':
        return np.hanning(size)
    else:  ## Raised Cosine
        x = np.linspace(-np.pi, np.pi, size)
        env = 0.5 + 0.5 * np.cos(x)
        fade_samples = min(size // 10, 32)
        for i in range(fade_samples):
            fade = i / fade_samples
            env[i] *= fade
            env[-(i+1)] *= fade
        return env

def generate_grain(source, start, duration, sample_rate, pitch=1.0):
    start_idx = int(start * sample_rate)
    grain_size = int(duration * sample_rate)

    if grain_size < 64:  
        grain_size = 64

    sample_indices = start_idx + np.arange(grain_size, dtype=np.float64) * pitch
    sample_indices = np.clip(sample_indices, 0, len(source) - 1)
    grain = np.interp(sample_indices, np.arange(len(source)), source)
    env = grain_envelope(grain_size,shape='smooth')  
    grain *= env
    return grain

## Applies processing depending on category (low,mid,high,rhythmic)

def category_processing(audio, sample_rate, category):
    # All processing functions now expect mono audio
    if category == 'rhythmic':  
        audio = rhythmic_quantization(audio, sample_rate)
        # Lighter compression
        comp = Compressor(threshold_db=-18, ratio=2.5, attack_ms=5, release_ms=80)
        audio = comp(audio, sample_rate)
        
        highpass = HighpassFilter(cutoff_frequency_hz=80)
        audio = highpass(audio, sample_rate)
        
        # Move distortion before the final lowpass to tame added highs afterward
        distortion = Distortion(drive_db=2.0)  
        audio = distortion(audio, sample_rate)
        
        lowpass = LowpassFilter(cutoff_frequency_hz=12000)
        audio = lowpass(audio, sample_rate)
       
        gain = Gain(gain_db=1.0)
        audio = gain(audio, sample_rate)
        
        return audio
    
    elif category == 'low':
        # Filter first so subsonics/highs don't drive compression
        highpass = HighpassFilter(cutoff_frequency_hz=25)
        audio = highpass(audio, sample_rate)
        
        lowpass = LowpassFilter(cutoff_frequency_hz=2500)
        audio = lowpass(audio, sample_rate)
        
        comp = Compressor(threshold_db=-25, ratio=2.0, attack_ms=80, release_ms=400)
        audio = comp(audio, sample_rate)
        
        chorus = Chorus(rate_hz=0.2, depth=0.08, mix=0.12)
        audio = chorus(audio, sample_rate)
        
        gain = Gain(gain_db=0.5)
        audio = gain(audio, sample_rate)
        
    elif category == 'mid':
        highpass = HighpassFilter(cutoff_frequency_hz=60)
        audio = highpass(audio, sample_rate)
        
        lowpass = LowpassFilter(cutoff_frequency_hz=10000)
        audio = lowpass(audio, sample_rate)
        
        comp = Compressor(threshold_db=-22, ratio=1.8, attack_ms=50, release_ms=250)
        audio = comp(audio, sample_rate)
        
        chorus = Chorus(rate_hz=0.3, depth=0.06, mix=0.12)
        audio = chorus(audio, sample_rate)
        
        gain = Gain(gain_db=0.3)
        audio = gain(audio, sample_rate)
        
    elif category == 'high':
        highpass = HighpassFilter(cutoff_frequency_hz=400)
        audio = highpass(audio, sample_rate)
        
        lowpass = LowpassFilter(cutoff_frequency_hz=16000)
        audio = lowpass(audio, sample_rate)
        
        comp = Compressor(threshold_db=-20, ratio=2.0, attack_ms=10, release_ms=120)
        audio = comp(audio, sample_rate)
        
        chorus = Chorus(rate_hz=0.4, depth=0.10, mix=0.12)
        audio = chorus(audio, sample_rate)
        
        gain = Gain(gain_db=-0.5)
        audio = gain(audio, sample_rate)
        
    return audio

def rhythmic_quantization(audio, sample_rate, bpm=120):
    """Apply rhythmic quantization to mono audio."""
    
    beat_length = 60.0 / bpm  
    samples_per_beat = int(beat_length * sample_rate)
    samples_per_sixteenth = samples_per_beat // 4  
    
    total_samples = len(audio)
    
    rhythmic_patterns = [
        [1.0, 0.3, 0.7, 0.2],  
        [1.0, 0.1, 0.8, 0.1], 
        [1.0, 0.5, 1.0, 0.3],  
        [1.0, 0.2, 0.6, 0.1],  
        [0.8, 0.1, 1.0, 0.4],  
    ]
    
    pattern = random.choice(rhythmic_patterns)
    step_count = (total_samples + samples_per_sixteenth - 1) // samples_per_sixteenth
    gate_probabilities = np.array([
        np.clip(
            pattern[index % len(pattern)] * random.uniform(0.8, 1.2),
            0.05,
            1.0,
        )
        for index in range(step_count)
    ], dtype=np.float32)
    sample_indices = np.arange(total_samples)
    step_indices = sample_indices // samples_per_sixteenth
    within_step = sample_indices % samples_per_sixteenth
    step_lengths = np.minimum(
        samples_per_sixteenth,
        total_samples - step_indices * samples_per_sixteenth,
    )
    gate_lengths = step_lengths // 8
    rhythm_envelope = 0.9 * gate_probabilities[step_indices]

    attack = (gate_lengths > 0) & (within_step < gate_lengths)
    attack_progress = np.zeros(total_samples, dtype=np.float32)
    attack_progress[attack] = within_step[attack] / gate_lengths[attack]
    rhythm_envelope[attack] = (
        attack_progress[attack] ** 2 * gate_probabilities[step_indices[attack]]
        + (1 - attack_progress[attack] ** 2) * 0.1
    )

    release = (gate_lengths > 0) & (within_step > step_lengths * 7 // 8)
    release_progress = np.zeros(total_samples, dtype=np.float32)
    release_progress[release] = (
        step_lengths[release] - within_step[release]
    ) / gate_lengths[release]
    rhythm_envelope[release] = (
        release_progress[release] ** 1.5 * gate_probabilities[step_indices[release]]
        + (1 - release_progress[release] ** 1.5) * 0.1
    )

    attack_lengths = np.minimum(step_lengths // 6, 512)
    punch = np.ones(total_samples, dtype=np.float32)
    punch_mask = (attack_lengths > 0) & (within_step < attack_lengths)
    punch[punch_mask] = np.maximum(
        1.0,
        1.3 * (1.0 - within_step[punch_mask] / attack_lengths[punch_mask]),
    )
    final_envelope = rhythm_envelope * punch
    
    # CHANGED: Apply to mono audio directly (no need for newaxis)
    return audio * final_envelope
    
def generate_audio_stream(
    params, duration, sources=None, category=None, target_sample_rate=None
):
    
    grain_duration, overlap, pitch_variance, start_offset_range, room_size, damping, wet_level, num_voices, low_cutoff_hz, high_cutoff_hz, ladder_resonance = params
    
    if high_cutoff_hz <= low_cutoff_hz:
        low_cutoff_hz, high_cutoff_hz = high_cutoff_hz, low_cutoff_hz
    
    if sources is None:
        audio_dir = os.path.join(DATA_DIR, "audio")
        if os.path.isdir(audio_dir):
            audio_files = [
                name
                for name in sorted(os.listdir(audio_dir))
                if name.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
            ]
            sources = [os.path.join(audio_dir, audio_files[0])] if audio_files else []
        else:
            sources = []

    loaded_sources = []
    sample_rate = target_sample_rate
    
    for source_path in sources:
        try:
            source, sr = sf.read(source_path, dtype="float32")
            if source.ndim > 1:
                source = np.mean(source, axis=1, dtype=np.float32)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                source = librosa.resample(
                    source, orig_sr=sr, target_sr=sample_rate
                ).astype(np.float32, copy=False)
            loaded_sources.append(source)
        except Exception as e:
            print(f"Warning: Could not load {source_path}: {e}")
            continue
    
    if not loaded_sources:
        raise FileNotFoundError("No readable audio sources were provided or found in data/audio.")
    
    if duration <= 0 or grain_duration <= 0 or overlap <= 0 or num_voices <= 0:
        raise ValueError("duration, grain_duration, overlap, and num_voices must be positive")

    buffer_size = int(sample_rate * duration)
    # CHANGED: Create mono buffer instead of stereo
    grain_buffer = np.zeros(buffer_size, dtype=np.float32)
    current_position = 0
    
    reverb = Reverb()
    reverb.room_size = room_size
    reverb.damping = damping
    reverb.wet_level = wet_level

    grain_period = grain_duration / overlap  
    start_offset_range = int(start_offset_range)

    total_grains = int(duration / grain_period)  
    
    total_grain_count = total_grains * int(num_voices)
    for grain_idx in range(total_grain_count):
        src_idx = random.randint(0, len(loaded_sources) - 1)
        source = loaded_sources[src_idx]

        start_offset = random.randint(0, min(start_offset_range, len(source) - 1))
        start_time = start_offset / sample_rate
        pitch = 1.0 + random.uniform(-pitch_variance, pitch_variance)
        grain = generate_grain(source, start_time, grain_duration, sample_rate, pitch)

        max_grain_val = np.max(np.abs(grain))
        if max_grain_val > 1e-10:
            grain = grain / max_grain_val

        grain_length = len(grain)

        fade_in_samples = min(grain_length // 2, 2048)
        fade_out_samples = min(grain_length // 2, 2048)
        fade_envelope = np.ones(grain_length)

        if fade_in_samples > 0:
            fade = np.sin((np.arange(fade_in_samples) / fade_in_samples) * (np.pi / 2)) ** 0.3
            fade_envelope[:fade_in_samples] = fade

        if fade_out_samples > 0:
            remaining = np.sin((np.arange(fade_out_samples, 0, -1) / fade_out_samples) * (np.pi / 2)) ** 0.3
            fade_envelope[-fade_out_samples:] = remaining

        scaled_grain = grain * fade_envelope * 0.4

        valid_len = min(grain_length, buffer_size - current_position)
        if valid_len > 0:
            indices = current_position + np.arange(valid_len, dtype=np.int64)
            grain_buffer[indices] += scaled_grain[:valid_len]

        current_position = (current_position + int(grain_length / overlap)) % buffer_size

    # CHANGED: Convert mono to stereo for reverb processing, then back to mono
    stereo_buffer = np.column_stack([grain_buffer, grain_buffer])  # Duplicate mono to stereo for reverb
    window_with_reverb = reverb.process(stereo_buffer, sample_rate)
    
    # Convert back to mono after reverb
    filtered = np.mean(window_with_reverb, axis=1)  # Average stereo back to mono

    # Apply filters to mono signal
    highpass = HighpassFilter(cutoff_frequency_hz=max(20, min(low_cutoff_hz, sample_rate/4)))
    filtered = highpass(filtered, sample_rate)

    lowpass = LowpassFilter(cutoff_frequency_hz=max(2000, min(high_cutoff_hz, sample_rate/2.1)))
    filtered = lowpass(filtered, sample_rate)

    ladder_cutoff = np.clip((low_cutoff_hz + high_cutoff_hz) / 2, 200, 8000)
    ladder = LadderFilter(cutoff_hz=ladder_cutoff, resonance=ladder_resonance)
    filtered = ladder(filtered, sample_rate)

    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    if category:
        filtered = category_processing(filtered, sample_rate, category)

    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(filtered))
    if peak > 0.99:
        filtered = filtered * (0.99 / peak)

    return filtered, sample_rate  # Returns mono audio

###Function to extract multiple audio descriptors from an audio signal

def extract_audio_descriptors(audio, sample_rate):
    n_fft = 2048
    hop_length = 512

   
    if audio.ndim == 1:
        descriptors = _extract_descriptors_mono(audio, sample_rate, n_fft, hop_length)
    else:
        # Fallback in case stereo somehow gets through
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        descriptors = _extract_descriptors_mono(audio_mono, sample_rate, n_fft, hop_length)
    
    return descriptors

def _extract_descriptors_mono(audio, sample_rate, n_fft, hop_length):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
  
    centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sample_rate))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sample_rate, roll_percent=0.85))
    
    # Spectral flux (rate of spectral change)
    spectral_flux = np.mean(np.sum(np.diff(S, axis=1)**2, axis=0)) if S.shape[1] > 1 else 0.0
    
    # Onset rate (onsets per second)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate, hop_length=hop_length)
    onset_rate = len(onset_frames) / (len(audio) / sample_rate)
    
    # Zero Crossing Rate for rhythmic/percussive detection
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=hop_length))
    
    # Return: [centroid, rolloff, spectral_flux, onset_rate, zcr]
    return np.array([centroid, rolloff, spectral_flux, onset_rate, zcr])