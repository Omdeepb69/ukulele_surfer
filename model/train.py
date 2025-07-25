import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from .cnn_model import UkuleleCNN
from .utils import AudioProcessor

class UkuleleDataset(Dataset):
    def __init__(self, audio_files, labels, processor, max_length=88200):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        audio, sr = librosa.load(audio_path, sr=self.processor.sample_rate)
        
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
            
        features = self.processor.preprocess_audio(audio)
        
        return features.squeeze(0), torch.tensor(label, dtype=torch.long)

class ModelTrainer:
    def __init__(self, data_dir='data', model_dir='model'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.processor = AudioProcessor()
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.label_to_idx = {
            'C_major': 0,
            'A_minor': 1, 
            'G_major': 2,
            'F_major': 3,
            'pluck_burst': 4
        }
        
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def scan_data_folders(self):
        """Automatically detect all folders in data directory and create labels"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found!")
            return {}
            
        found_labels = {}
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                audio_files = []
                for file in os.listdir(item_path):
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(item_path, file))
                
                if audio_files:
                    found_labels[item] = audio_files
                    
        return found_labels
        
    def load_dataset(self, auto_detect=True):
        audio_files = []
        labels = []
        
        if auto_detect:
            print("🔍 Auto-detecting audio files in data folders...")
            found_data = self.scan_data_folders()
            
            if not found_data:
                print("❌ No audio files found in data directory!")
                return [], []
                
            # Update label mappings based on found folders
            self.label_to_idx = {}
            self.idx_to_label = {}
            
            for idx, label_name in enumerate(sorted(found_data.keys())):
                self.label_to_idx[label_name] = idx
                self.idx_to_label[idx] = label_name
                
            print(f"📁 Found {len(found_data)} categories:")
            for label_name, files in found_data.items():
                print(f"   {label_name}: {len(files)} files")
                for file_path in files:
                    audio_files.append(file_path)
                    labels.append(self.label_to_idx[label_name])
                    
        else:
            # Original method - use predefined labels
            for label_name, label_idx in self.label_to_idx.items():
                label_dir = os.path.join(self.data_dir, label_name)
                if os.path.exists(label_dir):
                    for filename in os.listdir(label_dir):
                        if filename.endswith('.wav'):
                            audio_files.append(os.path.join(label_dir, filename))
                            labels.append(label_idx)
        
        return audio_files, labels
    
    def train(self, epochs=100, batch_size=16, learning_rate=0.001, auto_detect=True):
        print("Loading dataset...")
        audio_files, labels = self.load_dataset(auto_detect=auto_detect)
        
        if len(audio_files) == 0:
            print("❌ No training data found!")
            if auto_detect:
                print("💡 Make sure you have audio files in folders under 'data/' directory")
                print("   Example: data/C_major/chord1.wav, data/A_minor/chord2.wav")
            else:
                print("💡 Run recording mode first: python main.py --mode record --label C_major")
            return
            
        print(f"✅ Found {len(audio_files)} samples across {len(self.label_to_idx)} categories")
        
        # Check if we have enough samples per class
        samples_per_class = {}
        for label_name, label_idx in self.label_to_idx.items():
            count = labels.count(label_idx)
            samples_per_class[label_name] = count
            
        print("📊 Samples per category:")
        for label_name, count in samples_per_class.items():
            status = "✅" if count >= 20 else "⚠️" if count >= 10 else "❌"
            print(f"   {status} {label_name}: {count} samples")
            
        if any(count < 10 for count in samples_per_class.values()):
            print("⚠️  Warning: Some categories have very few samples. Consider adding more for better accuracy.")
        
        # Update model architecture to match number of classes
        num_classes = len(self.label_to_idx)

        # Check if we have enough samples for stratified split
        min_samples_per_class = min(samples_per_class.values())
        if min_samples_per_class < 2 or len(audio_files) < 2 * num_classes:
            print("⚠️  Not enough samples per class for stratified validation split. Using all data for training and skipping validation.")
            train_files, train_labels = audio_files, labels
            val_files, val_labels = [], []
        else:
            train_files, val_files, train_labels, val_labels = train_test_split(
                audio_files, labels, test_size=0.2, random_state=42, stratify=labels
            )
        
        train_dataset = UkuleleDataset(train_files, train_labels, self.processor)
        if val_files and val_labels:
            val_dataset = UkuleleDataset(val_files, val_labels, self.processor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_dataset = None
            val_loader = None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        model = UkuleleCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_acc = 0.0
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
            
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1)
                        val_correct += pred.eq(target).sum().item()
                
                train_acc = 100. * train_correct / len(train_dataset)
                val_acc = 100. * val_correct / len(val_dataset)
                
                scheduler.step(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'label_to_idx': self.label_to_idx,
                        'idx_to_label': self.idx_to_label,
                        'num_classes': num_classes
                    }, os.path.join(self.model_dir, 'ukulele_model.pth'))
                
                if epoch % 10 == 0:
                    print(f'📈 Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            else:
                train_acc = 100. * train_correct / len(train_dataset)
                if epoch % 10 == 0:
                    print(f'📈 Epoch {epoch}: Train Acc: {train_acc:.2f}% (no validation set)')
        
        print(f'🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        print(f'💾 Model saved to: {os.path.join(self.model_dir, "ukulele_model.pth")}')
        
        return best_val_acc