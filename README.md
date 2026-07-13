# Speech-To-Text

A lightweight, cross-platform, local speech-to-text dictation tool powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). 

It features two modes of operation and automatically optimizes the model size based on your operating system (Mac vs Linux). 

## Features
- **100% Offline**: Uses local whisper models so your voice never leaves your device.
- **Push-To-Talk**: Press and hold a hotkey anywhere on your system to transcribe seamlessly into your active window.
- **OS-Aware Optimization**: 
  - Uses `base.en` on macOS.
  - Uses `small.en` on Linux.
- **Zero Focus Stealing**: The floating visual indicator uses transparent window geometry to ensure your active window never loses focus.

## Prerequisites
- Python 3.10+
- `pip` or `uv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RangerVinven/Speech-To-Text.git
   cd Speech-To-Text
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

*Note: The first time you run the scripts, it will automatically download the Whisper model to your local cache (~145MB for base.en, ~460MB for small.en).*

## Usage

### 1. Push-To-Talk (`main.py`)
This is the primary dictation script. It runs in the background and waits for your hotkey.
```bash
python main.py
```
**Default Hotkey**: `Ctrl + Option + T` (Mac) / `Ctrl + Alt + T` (Linux). 

* Press and hold the hotkey to start recording. A "Listening..." indicator will appear in the top right.
* Speak your phrase.
* Release the keys. The script will instantly transcribe your audio and type the text directly into whichever application you are focused on.

> **⚠️ macOS Users**: You **must** grant your Terminal app "Accessibility" permissions in `System Settings -> Privacy & Security -> Accessibility`. If you don't, the script will be forcefully killed by macOS when it attempts to type the output.

### 2. Continuous Transcription (`continuous_test.py`)
A terminal-only version that listens continuously using Voice Activity Detection (VAD).
```bash
python continuous_test.py
```
* Simply speak into your microphone.
* Whenever you pause for 0.4 seconds, the script will instantly transcribe the previous sentence and print it directly into the terminal window.
