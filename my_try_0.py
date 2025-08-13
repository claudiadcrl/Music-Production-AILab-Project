# -*- coding: utf-8 -*-
"""
Audio-Visual Synthesizer V2 - Proper Threading Version
------------------------------------------------------
- Combines robust audio/visual system from final-nearly.py
- Uses proper threading structure from draft.py
- Same gesture controls and effects mapping
- Windows compatible with fallback options
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from pathlib import Path
import sys

# Audio imports with fallbacks
try:
    import pygame
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    print("Using pygame for audio")
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: pygame not available - running in visualization-only mode")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available - using generated test tones")

try:
    from pedalboard import Pedalboard, Reverb, Gain, Delay, PitchShift, Distortion
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("Warning: pedalboard not available - effects disabled")

# Global effects shared between threads (same as draft.py)
current_effects = {
    "gain":         0.0,    # dB
    "pitch":        0.0,    # semitones  
    "distortion":   0.0,    # drive dB
    "reverb":       0.25,   # room_size 0..1
    "delay":        0.25    # seconds
}
effects_lock = threading.Lock()

# Audio control
audio_running = threading.Event()
audio_error = threading.Event()

# --------------------------- Audio System ---------------------------
class SimpleEffects:
    """Simplified audio effects that don't cause access violations"""
    
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.delay_buffer = np.zeros((int(samplerate * 2), 2), dtype=np.float32)  # 2 second max delay
        self.delay_pos = 0
        
    def apply_gain(self, audio, gain_db):
        """Apply gain in dB"""
        if gain_db == 0:
            return audio
        gain_linear = 10.0 ** (gain_db / 20.0)
        return audio * gain_linear
    
    def apply_simple_delay(self, audio, delay_seconds, feedback=0.3):
        """Simple delay effect"""
        if delay_seconds <= 0.01:
            return audio
            
        delay_samples = int(delay_seconds * self.samplerate)
        delay_samples = min(delay_samples, len(self.delay_buffer) - 1)
        
        output = audio.copy()
        
        for i in range(len(audio)):
            # Get delayed sample
            delay_idx = (self.delay_pos - delay_samples) % len(self.delay_buffer)
            delayed_sample = self.delay_buffer[delay_idx] * feedback
            
            # Mix with current sample
            output[i] = audio[i] + delayed_sample * 0.5
            
            # Store current sample in delay buffer
            self.delay_buffer[self.delay_pos] = output[i]
            self.delay_pos = (self.delay_pos + 1) % len(self.delay_buffer)
            
        return output
    
    def apply_simple_distortion(self, audio, drive_db):
        """Simple distortion effect"""
        if drive_db <= 0:
            return audio
            
        drive = 1.0 + (drive_db / 10.0)
        distorted = np.tanh(audio * drive) / drive
        return distorted
    
    def process_audio(self, audio, effects_params):
        """Process audio with current effects"""
        try:
            # Apply gain
            processed = self.apply_gain(audio, effects_params.get("gain", 0.0))
            
            # Apply distortion  
            processed = self.apply_simple_distortion(processed, effects_params.get("distortion", 0.0))
            
            # Apply delay
            processed = self.apply_simple_delay(processed, effects_params.get("delay", 0.0))
            
            # Soft limiting
            processed = np.tanh(processed * 0.8) * 0.9
            
            return processed.astype(np.float32)
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            return audio

class AudioManager:
    def __init__(self):
        self.audio_data = None
        self.samplerate = 44100
        self.effects = SimpleEffects(self.samplerate)
        self.loop_position = 0
        self.is_playing = False
        
    def load_audio(self, filepath):
        """Load audio file or create test tone"""
        try:
            if filepath and Path(filepath).exists() and SOUNDFILE_AVAILABLE:
                data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
                if data.shape[1] == 1:  # Convert mono to stereo
                    data = np.repeat(data, 2, axis=1)
                self.audio_data = data
                self.samplerate = int(sr)
                print(f"Loaded: {filepath} ({sr} Hz, {data.shape[1]} ch, {len(data)/sr:.1f}s)")
                return True
            else:
                # Create test tone
                print("Creating test audio (sine wave chord)")
                duration = 8.0
                t = np.linspace(0, duration, int(self.samplerate * duration), False)
                # Musical chord: C4, E4, G4 (262, 330, 392 Hz)
                wave = (np.sin(262 * 2 * np.pi * t) * 0.3 + 
                       np.sin(330 * 2 * np.pi * t) * 0.2 + 
                       np.sin(392 * 2 * np.pi * t) * 0.2)
                self.audio_data = np.column_stack([wave, wave]).astype(np.float32)
                return True
                
        except Exception as e:
            print(f"Audio loading error: {e}")
            return False
    
    def get_audio_chunk(self, chunk_size):
        """Get next chunk of audio with looping"""
        if self.audio_data is None:
            return np.zeros((chunk_size, 2), dtype=np.float32)
            
        end_pos = self.loop_position + chunk_size
        
        if end_pos >= len(self.audio_data):
            # Handle looping
            chunk1 = self.audio_data[self.loop_position:]
            remaining = chunk_size - len(chunk1)
            chunk2 = self.audio_data[:remaining] if remaining > 0 else np.zeros((0, 2), dtype=np.float32)
            chunk = np.vstack([chunk1, chunk2]) if len(chunk2) > 0 else chunk1
            self.loop_position = remaining if remaining > 0 else 0
        else:
            chunk = self.audio_data[self.loop_position:end_pos]
            self.loop_position = end_pos
            
        # Ensure correct size
        if len(chunk) != chunk_size:
            if len(chunk) > chunk_size:
                chunk = chunk[:chunk_size]
            else:
                padding = np.zeros((chunk_size - len(chunk), 2), dtype=np.float32)
                chunk = np.vstack([chunk, padding])
                
        return chunk

# Audio thread function (same structure as draft.py)
def start_audio():
    """Audio thread function - runs continuously like draft.py"""
    if not AUDIO_AVAILABLE:
        print("Audio not available")
        return
        
    try:
        chunk_size = 2048  # Larger chunks for stability
        
        while audio_running.is_set():
            if not pygame.mixer.get_busy():
                # Generate next audio chunk
                chunk = audio_manager.get_audio_chunk(chunk_size)
                
                # Apply effects (same as draft.py - copy effects under lock)
                with effects_lock:
                    effects_copy = current_effects.copy()
                
                processed_chunk = audio_manager.effects.process_audio(chunk, effects_copy)
                
                # Convert to pygame format (16-bit integers)
                audio_int16 = (processed_chunk * 32767).astype(np.int16)
                
                # Create pygame sound and play
                try:
                    sound = pygame.sndarray.make_sound(audio_int16)
                    sound.play()
                except Exception as e:
                    print(f"Pygame playback error: {e}")
                    time.sleep(0.1)
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
    except Exception as e:
        print(f"Audio thread error: {e}")
        audio_error.set()

# --------------------------- Vision System (same as both files) ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint_list = [[4,2,1], [8,6,5], [12,10,9], [16,14,13], [20,18,17]]
up = [0,0,0,0,0]

def draw_finger_angles(image, landmrk, label):
    count = 0
    for i, joint in enumerate(joint_list):
        a = np.array([landmrk[joint[0]].x, landmrk[joint[0]].y])
        b = np.array([landmrk[joint[1]].x, landmrk[joint[1]].y])
        c = np.array([landmrk[joint[2]].x, landmrk[joint[2]].y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
            
        is_finger_up = angle >= 150

        if label == "Right":
            up[i] = 1 if is_finger_up else 0
        elif label == "Left" and is_finger_up:
            count += 1
    
    if label == 'Left':
        cv2.putText(image, f"Fingers Left: {count}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def angle_between(v1, v2):
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    return a2 - a1

def open_webcam():
    for idx in range(3):  # Try multiple indices
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "CAP_ANY"
                        print(f"[vision] Opened webcam {idx} with {backend_name}")
                        return cap
                cap.release()
            except:
                continue
    raise RuntimeError("Could not open any webcam")

# Initialize these at module level
prev_vectors = [None, None]
rotation_sums = [0.0, 0.0]

# --------------------------- Main Function ---------------------------
def main():
    global audio_manager
    
    print("=== Audio-Visual Synthesizer V2 Starting ===")
    
    # Initialize audio system
    audio_manager = AudioManager()
    
    # Audio file path - change this to your file or leave as None for test tone
    AUDIO_FILE = r"C:\Users\zosia\Downloads\Alesis-Fusion-Bass-Loop.wav"
    
    if not audio_manager.load_audio(AUDIO_FILE):
        print("Failed to load audio - using test tone")
    
    try:
        # Start audio thread (same as draft.py)
        if AUDIO_AVAILABLE:
            audio_running.set()
            audio_thread = threading.Thread(target=start_audio, daemon=True)
            audio_thread.start()
            print("Audio thread started")
        else:
            print("Running in visualization-only mode")
        
        # Open webcam
        cap = open_webcam()
        
        print("\nControls:")
        print("- Thumb only: Gain (±20 dB)")
        print("- Thumb + Index: Pitch (±12 semitones) [visualization only]")
        print("- Thumb + Index + Middle: Distortion (0-15 dB)")
        print("- Thumb + Index + Middle + Ring: Reverb (0.1-0.9) [visualization only]")
        print("- All five fingers: Delay (0.05-0.75s)")
        print("- Press 'q' to quit, 'r' to reset rotation")
        
        # Main vision processing loop
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                rotation_deg_display = 0.0
                
                # Access global variables
                global prev_vectors, rotation_sums
                
                if not results.multi_hand_landmarks:
                    prev_vectors = [None, None]

                if results.multi_hand_landmarks:
                    for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(216,255,87), thickness=2, circle_radius=2)
                        )
                        
                        label = handedness.classification[0].label
                        draw_finger_angles(image, hand_landmarks.landmark, label)

                        # Calculate rotation
                        lms = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
                        wrist = lms[0]
                        index_tip = lms[8]
                        curr_vec = index_tip - wrist

                        while len(prev_vectors) <= idx:
                            prev_vectors.append(None)
                            rotation_sums.append(0.0)

                        if prev_vectors[idx] is not None and np.linalg.norm(curr_vec) > 1e-6:
                            dtheta = angle_between(prev_vectors[idx], curr_vec)
                            if abs(dtheta) < np.pi/2:
                                rotation_sums[idx] += dtheta
                        prev_vectors[idx] = curr_vec

                    # Find control hand (prefer right)
                    control_idx = -1
                    for i, handedness in enumerate(results.multi_handedness):
                        if handedness.classification[0].label == "Right":
                            control_idx = i
                            break
                    if control_idx == -1 and results.multi_hand_landmarks:
                        control_idx = 0

                    if control_idx != -1:
                        rot_deg = float(np.degrees(rotation_sums[control_idx]))
                        rotation_deg_display = rot_deg
                        rot_norm = float(np.clip(rot_deg / 180.0, -1.0, 1.0))

                        # Apply effects based on finger combinations (SAME AS DRAFT.PY)
                        with effects_lock:
                            if up == [1,0,0,0,0]:  # Thumb only -> Gain
                                current_effects["gain"] = rot_norm * 20.0
                            elif up[:2] == [1,1] and sum(up[2:]) == 0:  # Thumb + Index -> Pitch
                                current_effects["pitch"] = rot_norm * 12.0
                            elif up[:3] == [1,1,1] and sum(up[3:]) == 0:  # Thumb + Index + Middle -> Distortion
                                current_effects["distortion"] = max(0.0, rot_norm) * 15.0
                            elif up[:4] == [1,1,1,1] and up[4] == 0:  # Four fingers -> Reverb
                                current_effects["reverb"] = float(np.interp(rot_norm, [-1, 1], [0.1, 0.9]))
                            elif up == [1,1,1,1,1]:  # All five -> Delay
                                current_effects["delay"] = float(np.interp(rot_norm, [-1, 1], [0.05, 0.75]))

                # Draw HUD
                cv2.rectangle(image, (0,0), (640, 140), (0,0,0), -1)
                cv2.putText(image, f"Right Hand: {up}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(image, f"Rotation: {rotation_deg_display:.1f}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Audio status
                if AUDIO_AVAILABLE:
                    audio_status = "Audio: Active" if audio_running.is_set() and not audio_error.is_set() else "Audio: Error"
                    color = (0,255,0) if "Active" in audio_status else (0,0,255)
                else:
                    audio_status = "Audio: Visualization Only"
                    color = (0,255,255)
                cv2.putText(image, audio_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                with effects_lock:
                    hud = (f"Gain:{current_effects['gain']:.1f}dB | "
                           f"Pitch:{current_effects['pitch']:.1f}st | "
                           f"Dist:{current_effects['distortion']:.1f}dB | "
                           f"Rev:{current_effects['reverb']:.2f} | "
                           f"Del:{current_effects['delay']:.2f}s")
                cv2.putText(image, hud, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (87, 255, 216), 1)
                
                # Instructions
                cv2.putText(image, "Press 'q' to quit, 'r' to reset rotation", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Audio-Visual Synthesizer V2", image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset rotation
                    rotation_sums = [0.0, 0.0]
                    prev_vectors = [None, None]
                    print("Rotation reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        audio_running.clear()
        
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.quit()
            except:
                pass
        
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        print("Cleanup complete.")

if __name__ == "__main__":
    main()