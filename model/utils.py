import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def preprocess_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.FloatTensor(audio_data)
        else:
            audio_tensor = audio_data
            
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        mel_spec = self.mel_transform(audio_tensor)
        mel_db = self.amplitude_to_db(mel_spec)
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        
        return mel_db.unsqueeze(0)
    
    def extract_pitch(self, audio_data, sr=None):
        if sr is None:
            sr = self.sample_rate
            
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, 
                                             hop_length=self.hop_length)
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                
        if pitch_values:
            return np.median(pitch_values)
        return 0.0
    
    def classify_by_pitch(self, pitch_hz):
        if pitch_hz < 50:
            return 'pluck_burst'
            
        note_frequencies = {
            'C': 261.63,
            'G': 392.00, 
            'E': 329.63,
            'A': 440.00,
            'F': 349.23
        }
        
        chord_patterns = {
            'C_major': ['C', 'E', 'G'],
            'A_minor': ['A', 'C', 'E'],
            'G_major': ['G', 'B', 'D'],
            'F_major': ['F', 'A', 'C']
        }
        
        closest_note = min(note_frequencies.items(), 
                          key=lambda x: abs(x[1] - pitch_hz))
        
        for chord, notes in chord_patterns.items():
            if closest_note[0] in notes:
                return chord
                
        return 'C_major'