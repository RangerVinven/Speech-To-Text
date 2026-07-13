import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
import tkinter as tk

MODEL_SIZE = "base.en" if sys.platform == 'darwin' else "small.en"
SAMPLE_RATE = 16000

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully.")

# Global State
is_recording = False
audio_queue = queue.Queue()
ui_queue = queue.Queue()
audio_data = []
stream = None

# PTT Combination Configuration
COMBINATION = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char('t')}
current_keys = set()

def get_base_key(key):
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        return keyboard.Key.ctrl
    if key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
        return keyboard.Key.alt
    if key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
        return keyboard.Key.cmd
    if key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
        return keyboard.Key.shift
    if hasattr(key, 'char') and key.char is not None:
        return keyboard.KeyCode.from_char(key.char.lower())
    return key

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
            
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback)
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
    segments, info = model.transcribe(audio_np, beam_size=1)
    text = "".join([segment.text for segment in segments]).strip()
    
    print(f"Transcribed: {text}")
    
    if text:
        # Send text to main thread to be typed safely
        ui_queue.put({"action": "type", "text": text})

def on_press(key):
    base_key = get_base_key(key)
    current_keys.add(base_key)
    if COMBINATION.issubset(current_keys):
        if not is_recording:
            start_recording()

def on_release(key):
    base_key = get_base_key(key)
    if base_key in current_keys:
        current_keys.remove(base_key)
    
    if is_recording and not COMBINATION.issubset(current_keys):
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
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    # Setup UI FIRST so macOS creates the NSApplication on the main thread safely
    root = setup_ui()
    
    # THEN start the keyboard listener in the background
    listener_thread = threading.Thread(target=run_listener, daemon=True)
    listener_thread.start()
    
    print(f"\n--- Speech to Text (Push-To-Talk) Ready ---")
    print(f"Press and HOLD <Ctrl> + <Option> + <T> to record.")
    print(f"Release to stop and transcribe.")
    print(f"-------------------------------------------\n")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
