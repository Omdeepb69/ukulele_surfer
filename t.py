import soundfile as sf 
import sounddevice as sd

data, samplerate = sf.read("ukulele-one-shot-f_F.wav")
sf.write('output.wav', data, int(48000*.2))
info = sf.info("output.wav")
print(f"Output File info: {info}")
print(data.shape, samplerate)
duration = 3
samplerate = 48000
recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float64')
print("Recording started")
sd.wait()
sf.write('sfoutput.wav', recording, samplerate)