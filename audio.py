import sounddevice as sd
import soundfile as sf
import numpy as np
from threading import Thread
import aupyom

# Load audio file
AUDIO_FILE = 'C:/Users/compu/Downloads/Alesis-Fusion-Bass-Loop.wav'
data, samplerate = sf.read(AUDIO_FILE, always_2d=True)

# Prepare aupyom Sound object
# Convert stereo to mono for pitch processing if needed (aupyom works on 1D arrays)
mono_data = data.mean(axis=1).astype(np.float32)

# Loop settings
loop_position = 0
blocksize = 1024  # 23 ms latency

def audio_callback(outdata, frames, time, status):
    global loop_position
    if status:
        print(status)

    # Looping logic
    end_position = loop_position + frames
    if end_position >= len(mono_data):
        chunk = np.concatenate([
            mono_data[loop_position:],
            mono_data[:end_position % len(mono_data)]
        ])
        loop_position = end_position % len(mono_data)
    else:
        chunk = mono_data[loop_position:end_position]
        loop_position = end_position

    sound = aupyom.Sound(chunk, samplerate)
    sound.pitch_shift=1
    outdata[:] = sound

# Start audio stream
with sd.OutputStream(channels=2, callback=audio_callback,
                     blocksize=blocksize, samplerate=samplerate):
    print("Streaming with aupyom pitch shift... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping.")
