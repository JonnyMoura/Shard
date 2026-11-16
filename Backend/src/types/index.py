class AudioSolution:
    def __init__(self, parameters, audio_data):
        self.parameters = parameters  # Parameters used to generate the audio
        self.audio_data = audio_data  # The generated audio data

    def get_parameters(self):
        return self.parameters

    def get_audio_data(self):
        return self.audio_data


class AudioParameters:
    def __init__(
        self,
        grain_duration,
        overlap,
        pitch_variance,
        start_offset_range,
        room_size,
        damping,
        wet_level,
        num_voices,
        low_cutoff_hz,
        high_cutoff_hz,
        ladder_resonance,
        category=None
    ):
        self.grain_duration = grain_duration          # Duration of each grain in seconds
        self.overlap = overlap                        # Overlap factor between grains
        self.pitch_variance = pitch_variance          # Pitch variance for grains
        self.start_offset_range = start_offset_range  # Range of random start offsets for grains
        self.room_size = room_size                    # Reverb room size
        self.damping = damping                        # Reverb damping
        self.wet_level = wet_level                    # Reverb wet level
        self.num_voices = num_voices                  # Number of voices for grain synthesis
        self.low_cutoff_hz = low_cutoff_hz            # Lowpass filter cutoff frequency (Hz)
        self.high_cutoff_hz = high_cutoff_hz          # Highpass filter cutoff frequency (Hz)
        self.ladder_resonance = ladder_resonance      # Ladder filter resonance
        self.category = category                      # Category of the audio (e.g., 'low', 'mid', 'high', 'rhythmic')

    def to_dict(self):
        return {
            "grain_duration": self.grain_duration,
            "overlap": self.overlap,
            "pitch_variance": self.pitch_variance,
            "start_offset_range": self.start_offset_range,
            "room_size": self.room_size,
            "damping": self.damping,
            "wet_level": self.wet_level,
            "num_voices": self.num_voices,
            "low_cutoff_hz": self.low_cutoff_hz,
            "high_cutoff_hz": self.high_cutoff_hz,
            "ladder_resonance": self.ladder_resonance
        }
    

  
    


