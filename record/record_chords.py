import sounddevice as sd
import numpy as np
import os
import soundfile as sf
from datetime import datetime

class ChordRecorder:
    def __init__(self, sample_rate=44100, data_dir='data'):
        self.sample_rate = sample_rate
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        self.labels = ['C_major', 'A_minor', 'G_major', 'F_major', 'pluck_burst']
        for label in self.labels:
            os.makedirs(os.path.join(data_dir, label), exist_ok=True)
    
    def record_samples(self, label, num_samples, duration):
        if label not in self.labels:
            print(f"Invalid label. Choose from: {self.labels}")
            return
            
        print(f"Recording {num_samples} samples of {label} ({duration}s each)")
        print("Press Enter to start each recording...")
        
        for i in range(num_samples):
            input(f"Sample {i+1}/{num_samples} - Press Enter to record...")
            
            print("Recording... 3")
            sd.sleep(1000)
            print("2")
            sd.sleep(1000)
            print("1")
            sd.sleep(1000)
            print("GO!")
            
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, dtype='float32')
            sd.wait()
            print(f"Audio debug: min={audio.min()}, max={audio.max()}, mean={audio.mean()}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{timestamp}_{i+1:03d}.wav"
            filepath = os.path.join(self.data_dir, label, filename)
            
            sf.write(filepath, audio.flatten(), self.sample_rate)
            print(f"Saved: {filename}")
            
        print(f"Completed recording {num_samples} samples for {label}")
    
    def get_dataset_info(self):
        info = {}
        for label in self.labels:
            label_dir = os.path.join(self.data_dir, label)
            if os.path.exists(label_dir):
                files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
                info[label] = len(files)
            else:
                info[label] = 0
        return info