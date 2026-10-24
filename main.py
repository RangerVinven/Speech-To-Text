import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
import tkinter as tk

MODEL_SIZE = "large-v3-turbo"
SAMPLE_RATE = 16000

# Try GPU (CUDA) first with int8_float16 (optimal performance & lower VRAM on GTX 1660)
# Fallback gracefully to CPU if CUDA is unavailable or encounters an error.
try:
    print(f"Attempting to load Whisper model ({MODEL_SIZE}) on GPU (CUDA)...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
    print("Model loaded successfully on GPU (CUDA: int8_float16).")
except Exception as cuda_err:
    print(f"CUDA initialization failed ({cuda_err}). Falling back to CPU...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model loaded successfully on CPU (int8).")

# Global State
is_recording = False
audio_queue = queue.Queue()
ui_queue = queue.Queue()
audio_data = []
stream = None

# PTT Combination Configuration
current_keys = set()
listener = None

def is_combination_pressed(keys):
    for k in keys:
        if k == keyboard.Key.f9:
            return True
        if hasattr(k, 'name') and k.name == 'f9':
            return True
    return False

def get_webcam_device():
    """Finds the webcam (C920) device index, strictly excluding any Soundcore/Bluetooth devices."""
    try:
        devices = sd.query_devices()
        # 1. Look specifically for C920 / Webcam
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                if 'c920' in name or 'webcam' in name:
                    return idx
        # 2. Fallback: Any input device that is strictly NOT Soundcore / Bluetooth
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                if 'soundcore' not in name and 'bluez' not in name and 'q45' not in name:
                    return idx
    except Exception as e:
        print(f"Error querying audio devices: {e}", file=sys.stderr)
    return None

def start_recording():
    global is_recording, audio_data, stream
    if is_recording: return
    is_recording = True
    audio_data = []
    print("Recording started...")
    
    ui_queue.put("show")
    
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if is_recording:
            audio_queue.put(indata.copy())
            
    dev_idx = get_webcam_device()
    if dev_idx is not None:
        dev_name = sd.query_devices(dev_idx)['name']
        print(f"Using audio device: [{dev_idx}] {dev_name}")
    else:
        print("Warning: Webcam not found; using system default (non-Bluetooth fallback).", file=sys.stderr)

    stream = sd.InputStream(device=dev_idx, samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback)
    stream.start()

def stop_recording_and_transcribe():
    global is_recording, stream, audio_data
    if not is_recording: return
    is_recording = False
    print("Recording stopped. Transcribing...")
    
    ui_queue.put("hide")
    
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
        
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
        
    if not audio_data:
        print("No audio recorded.")
        return
        
    audio_np = np.concatenate(audio_data, axis=0).flatten()
    try:
        segments, info = model.transcribe(
            audio_np,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )
        text = "".join([segment.text for segment in segments]).strip()
    except Exception as e:
        print(f"Transcription error: {e}", file=sys.stderr)
        text = ""
    
    print(f"Transcribed: {text}")
    
    if text:
        # Send text to main thread to be typed safely
        ui_queue.put({"action": "type", "text": text})

def on_press(key):
    current_keys.add(key)
    if is_combination_pressed(current_keys):
        if not is_recording:
            start_recording()

def on_release(key):
    try:
        current_keys.remove(key)
    except KeyError:
        pass
    
    if is_recording and not is_combination_pressed(current_keys):
        # Run transcription in a background thread to keep keyboard listener fast
        threading.Thread(target=stop_recording_and_transcribe, daemon=True).start()

def execute_typing(text):
    print("\n[Attempting to type text into your active window...]")
    print("[If the script instantly crashes or aborts here, macOS blocked it!]")
    print("[You must grant Accessibility permissions in System Settings.]")
    
    # Initialize lazily to prevent startup aborts
    controller = keyboard.Controller()
    controller.type(text + " ")
    print("[Typing successful!]\n")

def check_ui_queue(root, x, y):
    while not ui_queue.empty():
        msg = ui_queue.get()
        if isinstance(msg, dict) and msg.get("action") == "type":
            root.after(100, lambda text=msg["text"]: execute_typing(text))
        elif msg == "show":
            # Move on-screen and fade in
            root.geometry(f"+{x}+{y}")
            root.attributes('-alpha', 0.85)
        elif msg == "hide":
            # Make fully invisible and move far off bottom-right
            root.attributes('-alpha', 0.0)
            root.geometry(f"+20000+20000")
    root.after(50, lambda: check_ui_queue(root, x, y))

def setup_ui():
    root = tk.Tk()
    root.title("Listening Indicator")
    
    root.overrideredirect(True)
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.85)
    root.config(bg='#282c34')
    
    label = tk.Label(root, text="🎙️ Listening...", font=("Helvetica", 18, "bold"), fg="#98c379", bg="#282c34")
    label.pack(padx=20, pady=10)
    
    window_width = 200
    window_height = 50
    screen_width = root.winfo_screenwidth()
    x = screen_width - window_width - 40
    y = 40
    
    # Start completely invisible and off bottom-right
    root.attributes('-alpha', 0.0)
    root.geometry(f"{window_width}x{window_height}+20000+20000")
    
    root.after(50, lambda: check_ui_queue(root, x, y))
    return root

def run_listener():
    global listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        listener = l
        l.join()

if __name__ == "__main__":
    # Setup UI FIRST so macOS creates the NSApplication on the main thread safely
    root = setup_ui()
    
    # THEN start the keyboard listener in the background
    listener_thread = threading.Thread(target=run_listener, daemon=True)
    listener_thread.start()
    
    print(f"\n--- Speech to Text (Push-To-Talk) Ready ---")
    print(f"Press and HOLD <F9> to record.")
    print(f"Release to stop and transcribe.")
    print(f"-------------------------------------------\n")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
