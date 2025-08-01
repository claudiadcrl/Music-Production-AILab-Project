import librosa
import sounddevice as sd
import threading
import time

# Load audio
y, sr = librosa.load('C:/Users/compu/Downloads/Alesis-Fusion-Bass-Loop.wav', sr=44100)
segment = y[:10 * sr]  # First 10 seconds

def loop_audio():
    while True:
        sd.play(segment, samplerate=sr)
        sd.wait()

# Start audio loop in a background thread
thread = threading.Thread(target=loop_audio, daemon=True)
thread.start()

# Main thread stays alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")
#stops with ctrl+c