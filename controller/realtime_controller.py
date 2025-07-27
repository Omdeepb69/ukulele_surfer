import sounddevice as sd
import numpy as np
import torch
import pyautogui
import time
import threading
from collections import deque
from model.cnn_model import UkuleleCNN
from model.utils import AudioProcessor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealTimeController:
    def __init__(self, detection_mode='model', sample_rate=44100, 
                 buffer_duration=2.0, confidence_threshold=0.7, debug=False):
        self.detection_mode = detection_mode
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        
        self.processor = AudioProcessor(sample_rate)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        self.action_mapping = {
            'C_major': 'up',
            'A_minor': 'down', 
            'G_major': 'left',
            'F_major': 'right',
            'pluck_burst': 'space'
        }
        
        self.last_action_time = 0
        self.min_action_interval = 0.3
        
        self.running = False
        
        if detection_mode == 'model':
            self._load_model()
        
        if debug:
            self._setup_debug_visualization()
            
        pyautogui.FAILSAFE = False
        
    def _load_model(self):
        try:
            checkpoint = torch.load('model/ukulele_model.pth', map_location='cpu')
            num_classes = checkpoint.get('num_classes', 5)
            self.model = UkuleleCNN(num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.idx_to_label = checkpoint['idx_to_label']
            self.label_to_idx = checkpoint['label_to_idx']
            
            # Update action mapping based on loaded labels
            default_actions = ['up', 'down', 'left', 'right', 'space']
            self.action_mapping = {}
            
            for label, idx in self.label_to_idx.items():
                if idx < len(default_actions):
                    self.action_mapping[label] = default_actions[idx]
                else:
                    # For additional labels beyond the default 5
                    self.action_mapping[label] = 'space'
                    
            print(f"✅ Model loaded successfully with {num_classes} classes")
            print(f"🎮 Action mappings: {self.action_mapping}")
            
        except FileNotFoundError:
            print("❌ Model not found. Train the model first or use --detection pitch")
            self.detection_mode = 'pitch'
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.detection_mode = 'pitch'
            
    def _setup_debug_visualization(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.ax1.set_title('Audio Waveform')
        self.ax2.set_title('Predictions')
        self.prediction_history = deque(maxlen=100)
        
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        
        self.audio_buffer.extend(indata[:, 0])
        
    def _predict_chord(self, audio_data):
        if self.detection_mode == 'model' and hasattr(self, 'model'):
            try:
                features = self.processor.preprocess_audio(audio_data)
                
                with torch.no_grad():
                    output = self.model(features)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predicted_label = self.idx_to_label[predicted.item()]
                    return predicted_label, confidence.item()
            except Exception as e:
                print(f"Model prediction error: {e}")
                return None, 0.0
                
        elif self.detection_mode == 'pitch':
            try:
                pitch = self.processor.extract_pitch(audio_data)
                predicted_label = self.processor.classify_by_pitch(pitch)
                return predicted_label, 0.8
            except Exception as e:
                print(f"Pitch detection error: {e}")
                return None, 0.0
            
        return None, 0.0
    
    def _execute_action(self, action):
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval:
            return
            
        if action in ['up', 'down', 'left', 'right', 'space']:
            pyautogui.press(action)
            print(f"🎮 Action executed: {action}")
            self.last_action_time = current_time
            
    def _processing_loop(self):
        print("🎧 Audio processing loop started - listening for any sound...")
        
        while self.running:
            # Process audio more frequently with smaller chunks
            if len(self.audio_buffer) >= 1024:  # Much smaller buffer for faster detection
                # Get recent audio data
                recent_audio = np.array(list(self.audio_buffer)[-8192:])  # Last ~0.2 seconds
                
                # Calculate volume levels
                rms = np.sqrt(np.mean(recent_audio**2))
                max_amplitude = np.max(np.abs(recent_audio))
                
                # Print sound detection info - much lower threshold
                if rms > 0.001:  # Very sensitive threshold
                    print(f"🔊 SOUND DETECTED! RMS: {rms:.4f}, Max: {max_amplitude:.4f}")
                    
                    # Try to predict only if there's significant sound
                    if rms > 0.01:
                        print("🎵 Analyzing audio for chord recognition...")
                        prediction, confidence = self._predict_chord(recent_audio)
                        
                        if prediction:
                            if confidence > self.confidence_threshold:
                                print(f"✅ RECOGNIZED: {prediction} (confidence: {confidence:.2f})")
                                if prediction in self.action_mapping:
                                    action = self.action_mapping[prediction]
                                    self._execute_action(action)
                                else:
                                    print(f"⚠️  No action mapped for: {prediction}")
                            else:
                                print(f"❓ LOW CONFIDENCE: {prediction} (confidence: {confidence:.2f}) - threshold is {self.confidence_threshold}")
                        else:
                            print("❌ NOT RECOGNIZED - sound detected but no chord identified")
                            
                        if self.debug:
                            self.prediction_history.append((prediction or "unknown", confidence))
                            self._update_debug_plot(recent_audio)
                    else:
                        print(f"🔇 Sound too quiet for analysis (RMS: {rms:.4f}, need > 0.01)")
                
                # Clear some buffer to prevent memory issues but keep recent data
                if len(self.audio_buffer) > self.buffer_size * 0.8:
                    # Remove older data, keep recent
                    for _ in range(1024):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
            time.sleep(0.01)  # Much faster processing
    
    def _update_debug_plot(self, audio_data):
        try:
            self.ax1.clear()
            self.ax1.plot(audio_data)
            self.ax1.set_title('Audio Waveform')
            
            if self.prediction_history:
                recent_predictions = list(self.prediction_history)[-10:]
                labels = [p[0] for p in recent_predictions]
                confidences = [p[1] for p in recent_predictions]
                
                self.ax2.clear()
                self.ax2.bar(range(len(labels)), confidences)
                self.ax2.set_xticks(range(len(labels)))
                self.ax2.set_xticklabels(labels, rotation=45)
                self.ax2.set_title('Recent Predictions')
                
            plt.tight_layout()
            plt.pause(0.01)
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start(self):
        print("🎸 Starting Ukulele Subway Surfers Controller...")
        print("Game controls:")
        print("- C Major: Jump (↑)")
        print("- A Minor: Slide (↓)")
        print("- G Major: Move Left (←)")
        print("- F Major: Move Right (→)")
        print("- Pluck Burst: Hoverboard (Space)")
        print(f"\n🎚️  Audio settings:")
        print(f"- Sample rate: {self.sample_rate} Hz")
        print(f"- Detection mode: {self.detection_mode}")
        print(f"- Confidence threshold: {self.confidence_threshold}")
        print("\n🔍 Debug mode: Will print ANY sound detection")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        
        processing_thread = threading.Thread(target=self._processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        try:
            print("🎤 Opening audio stream...")
            with sd.InputStream(callback=self._audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=1024):
                print("✅ Audio stream active - make some noise!")
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping controller...")
            self.running = False
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            self.running = False
            
        if self.debug:
            plt.close('all')


# Add this at the bottom to test basic functionality
if __name__ == "__main__":
    print("🎸 Testing Ukulele Controller with enhanced sound detection")
    print("📋 Available audio devices:")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"Error querying devices: {e}")
    
    controller = RealTimeController(detection_mode='pitch', debug=False)
    controller.start()