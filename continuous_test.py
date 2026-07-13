import sys
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "base.en" if sys.platform == 'darwin' else "small.en"
SAMPLE_RATE = 16000

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully.")

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

print("\n🎙️  Listening continuously... Speak into the microphone. (Press Ctrl+C to stop)")
print("-------------------------------------------------------------------------")

# Basic Voice Activity Detection parameters
ENERGY_THRESHOLD = 0.005  # Volume threshold to detect speech
SILENCE_DURATION = 0.4    # Reduced to 0.4 for much faster response!
BLOCK_DURATION = 0.1      # Size of each audio block in seconds
block_size = int(SAMPLE_RATE * BLOCK_DURATION)

buffer = []
silent_blocks = 0
is_speaking = False

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=block_size, callback=callback):
        while True:
            chunk = audio_queue.get()
            energy = np.sqrt(np.mean(chunk**2))
            
            if energy > ENERGY_THRESHOLD:
                is_speaking = True
                silent_blocks = 0
                buffer.append(chunk)
            elif is_speaking:
                silent_blocks += 1
                buffer.append(chunk)
                
                # If we've had enough silence, assume the sentence/phrase is done and transcribe it
                if silent_blocks > (SILENCE_DURATION / BLOCK_DURATION):
                    audio_np = np.concatenate(buffer, axis=0).flatten()
                    
                    # Only transcribe if we captured at least a little bit of audio
                    if len(audio_np) > SAMPLE_RATE * 0.3:
                        # Show a little processing indicator
                        print("...", end="\r", flush=True)
                        
                        segments, _ = model.transcribe(audio_np, beam_size=1)
                        text = " ".join([segment.text.strip() for segment in segments]).strip()
                        
                        if text:
                            # \033[K clears the "..." indicator on the line
                            print(f"\033[K> {text}")
                            
                    # Reset state for the next phrase
                    buffer = []
                    is_speaking = False
                    silent_blocks = 0
                    
except KeyboardInterrupt:
    print("\n\nStopped listening.")
