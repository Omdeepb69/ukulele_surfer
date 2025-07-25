from TTS.api import TTS

# Init TTS with a model that works on CPU
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Run TTS
text = "Hey nibba, this is CPU-based voice cloning. Slow but sexy."
tts.tts_to_file(text=text, file_path="output.wav")
