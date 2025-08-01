import sounddevice as sd
import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Reverb, Gain, Delay, PitchShift, Distortion
from threading import Thread

#RIGHT HAND
#Since the anchor is the wrist, we can have a parameter also for only the thumb
#Thumb: Gain
#Thumb+Index: Pitch
#Thumb+Index+Middle: Distorsion
#Thumb+Index+Middle+Ring: Reverb
#All five: Delay

#LEFT HAND
#Five tracks, one per finger

#Must stop if put away hands

# Load audio file
AUDIO_FILE = 'C:/Users/compu/Downloads/Alesis-Fusion-Bass-Loop.wav'
data, samplerate = sf.read(AUDIO_FILE, always_2d=True)

# Loop settings
loop_position = 0
blocksize = 1024 #23 ms latency

# Define your effect chain
board = Pedalboard([
    Gain(),
    PitchShift(),
    Distortion(),
    Reverb(room_size=0.25),
    Delay(delay_seconds=0.25)
])

def audio_callback(outdata, frames, time, status):
    global loop_position
    if status:
        print(status)

    # Looping logic
    end_position = loop_position + frames
    if end_position >= len(data):
        chunk = np.vstack([
            data[loop_position:], #Plays till the end of file
            data[:end_position % len(data)] #Wraps around and plays from the start of the file up to the amount needed to complete the audio block 
        ])
        loop_position = end_position % len(data)
    else:
        chunk = data[loop_position:end_position]
        loop_position = end_position

    # Apply effects
    effected = board(chunk, samplerate)
    outdata[:] = effected

# Start audio stream
with sd.OutputStream(channels=2, callback=audio_callback,
                     blocksize=blocksize, samplerate=samplerate):
    print("Streaming with effects... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping.")