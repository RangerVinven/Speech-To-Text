import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import time

MODEL_SIZE = "tiny.en"
SAMPLE_RATE = 16000
DURATION = 5  # record for 5 seconds

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully.")

print(f"\n🎙️  Recording for {DURATION} seconds... Speak now!")

# Record audio
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  # Wait until recording is finished

print("Recording finished. Transcribing...")

# Flatten the array for faster-whisper
audio_np = audio_data.flatten()

# Transcribe
segments, info = model.transcribe(audio_np, beam_size=1)
text = "".join([segment.text for segment in segments]).strip()

print("\n--- Transcription Result ---")
print(f'"{text}"')
print("----------------------------\n")
