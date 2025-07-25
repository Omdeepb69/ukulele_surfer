import torchaudio
import sounddevice as sd
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

waveform, sample_rate = torchaudio.load("ukulele-one-shot-f_F.wav")

mel = MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=128)(waveform)
db_mel = AmplitudeToDB()(mel)

print(db_mel.shape)

class SoundCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*61*13,128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            
        )
        
    def forward(self, x):
        return self.net(x.unsqueeze(1))
    
class AudioDataset(Dataset):
    def __init__(self, file_paths,labels):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=128),
            AmplitudeToDB()
        )
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        x,sr = torchaudio.load(self.file_paths[idx])
        x = self.mel_transform(x)
        return x, self.labels[idx]
    
train_loader = DataLoader(AudioDataset(file_paths,lables), batch_size=16, shuffle=True)

model = SoundCNN(num_classes=5) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in train_loader:
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
def listen_and_predict(model):
    recording = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    waveform = torch.tensor(recording).transpose(0, 1)
    mel = MelSpectrogram()(waveform)
    db_mel = AmplitudeToDB()(mel)
    db_mel = db_mel.unsqueeze(0) 
    output = model(db)
    pred = torch.argmax(output, dim=1)
    print(f"Predicted class: {pred.item()}")
    return pred.item()
    