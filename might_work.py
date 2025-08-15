# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from pathlib import Path


try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("Using sounddevice for audio")
except Exception:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available - running in visualization-only mode")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available - using generated test tones")


current_effects = {
    "gain_db":       0.0,   
    "bitcrush_amt":  0.0,   
    "clip_thresh":   1.0,   
    "drive_db":      0.0,   
    "invert_mix":    0.0,   
}

effects_lock = threading.Lock()

audio_running = threading.Event()
audio_error = threading.Event()


class AudioManager:
    def __init__(self):
        self.audio_data = None
        self.samplerate = 44100
        self.loop_position = 0
        self.blocksize = 1024
        self.is_playing = False

        
        self._bc_hold = None  
        self._bc_count = 0

    
    def load_audio(self, filepath):
        """Load audio file (stereo float32) or create a test tone."""
        try:
            if filepath and Path(filepath).exists() and SOUNDFILE_AVAILABLE:
                data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                self.audio_data = data
                self.samplerate = int(sr)
                print(f"Loaded: {filepath} ({sr} Hz, {data.shape[1]} ch, {len(data)/sr:.1f}s)")
                return True
            else:
                print("Creating test audio (sine wave chord)")
                duration = 8.0
                t = np.linspace(0, duration, int(self.samplerate * duration), False)
                wave = (np.sin(262 * 2 * np.pi * t) * 0.3 +
                        np.sin(330 * 2 * np.pi * t) * 0.2 +
                        np.sin(392 * 2 * np.pi * t) * 0.2).astype(np.float32)
                self.audio_data = np.column_stack([wave, wave]).astype(np.float32)
                return True
        except Exception as e:
            print(f"Audio loading error: {e}")
            return False

    
    def get_audio_chunk(self, frames):
        if self.audio_data is None:
            return np.zeros((frames, 2), dtype=np.float32)
        end_position = self.loop_position + frames
        if end_position >= len(self.audio_data):
            chunk = np.vstack([
                self.audio_data[self.loop_position:],
                self.audio_data[:end_position % len(self.audio_data)]
            ])
            self.loop_position = end_position % len(self.audio_data)
        else:
            chunk = self.audio_data[self.loop_position:end_position]
            self.loop_position = end_position
        return chunk

    
    @staticmethod
    def db_to_lin(db):
        return 10.0 ** (db / 20.0)

    def bitcrush(self, x, amt):
        """Bitcrusher with combined bit-depth + sample-rate reduction.
        amt 0.0 = clean, 1.0 = heavy.
        - Bits: 16 → 4 as amt goes 0→1
        - Rate reduction factor: 1 → 12 as amt goes 0→1 (sample-hold)
        """
        if amt <= 1e-4:
            return x

        
        bits = int(np.round(16 - amt * 12))  
        bits = max(4, min(16, bits))
        q_levels = (2 ** bits) - 1

        down = int(np.round(1 + amt * 11))  
        down = max(1, min(12, down))

        
        xq = np.clip(x, -1.0, 1.0)
        xq = np.round((xq * 0.5 + 0.5) * q_levels) / q_levels
        xq = (xq - 0.5) * 2.0

        if down == 1:
            return xq

        
        n, ch = xq.shape
        if self._bc_hold is None:
            self._bc_hold = np.zeros(ch, dtype=np.float32)
            self._bc_count = 0
        out = np.empty_like(xq)
        cnt = self._bc_count
        hold = self._bc_hold.copy()
        for i in range(n):
            if cnt == 0:
                hold = xq[i]
            out[i] = hold
            cnt = (cnt + 1) % down
        self._bc_count = cnt
        self._bc_hold = hold
        return out

    def distort(self, x, drive_db):
        if drive_db <= 1e-4:
            return x
        g = self.db_to_lin(drive_db)
        
        return np.tanh(g * x)

    @staticmethod
    def hard_clip(x, thresh):
        t = float(max(0.1, min(1.0, thresh)))
        return np.clip(x, -t, t)

    @staticmethod
    def invert_mix(x, mix):
        
        return (1.0 - 2.0 * float(max(0.0, min(1.0, mix)))) * x

    
    def audio_callback(self, outdata, frames, time_info, status):
        try:
            chunk = self.get_audio_chunk(frames).astype(np.float32)
            with effects_lock:
                fx = current_effects.copy()

            # Gain
            if abs(fx["gain_db"]) > 1e-4:
                chunk = chunk * self.db_to_lin(fx["gain_db"])

            # Bitcrush (bit depth + rate reduction)
            chunk = self.bitcrush(chunk, fx["bitcrush_amt"])

            # Distortion (waveshaper)
            chunk = self.distort(chunk, fx["drive_db"])

            # Hard clip
            chunk = self.hard_clip(chunk, fx["clip_thresh"])

            # Invert mix
            chunk = self.invert_mix(chunk, fx["invert_mix"])

            # Prevent NaNs/inf
            np.nan_to_num(chunk, copy=False)
            outdata[:] = chunk
        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata.fill(0)




audio_manager = None

def start_audio():
    if not SOUNDDEVICE_AVAILABLE:
        print("SoundDevice not available")
        audio_error.set()
        return
    try:
        print(f"Starting audio stream: {audio_manager.samplerate} Hz, blocksize {audio_manager.blocksize}")
        with sd.OutputStream(
            channels=2,
            callback=audio_manager.audio_callback,
            blocksize=audio_manager.blocksize,
            samplerate=audio_manager.samplerate,
            dtype=np.float32,
        ):
            while audio_running.is_set():
                time.sleep(0.1)
        print("Audio stream stopped")
    except Exception as e:
        print(f"Audio thread error: {e}")
        audio_error.set()



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
    cap=cv2.VideoCapture(0)
    return cap


prev_vectors = [None, None]
rotation_sums = [0.0, 0.0]


def main():
    global audio_manager
    print("=== Audio-Visual Synthesizer V3 - Custom FX ===")

    audio_manager = AudioManager()

    # Change to your path or leave empty to use the built-in test tone
    AUDIO_FILE = r"C:\\Users\\zosia\\Downloads\\Alesis-Fusion-Bass-Loop.wav"

    if not audio_manager.load_audio(AUDIO_FILE):
        print("Failed to load audio - using test tone")

    try:
        if SOUNDDEVICE_AVAILABLE:
            audio_running.set()
            audio_thread = threading.Thread(target=start_audio, daemon=True)
            audio_thread.start()
            print("Audio thread started")
            time.sleep(0.1)
            if audio_error.is_set():
                print("Audio failed to start")
            else:
                print("Audio system ready")
        else:
            print("Running in visualization-only mode")

        cap = open_webcam()

        print("\nControls:")
        print("- Thumb only: Gain (±20 dB)")
        print("- Thumb + Index: Bitcrush (0→1)")
        print("- Thumb + Index + Middle: Clip threshold (1.0→0.1)")
        print("- Thumb + Index + Middle + Ring: Distortion drive (0→24 dB)")
        print("- All five fingers: Invert mix (0→1)")
        print("- Press 'q' to quit, 'r' to reset rotation")

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                rotation_deg_display = 0.0

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

                    # Choose right hand if present, else first hand
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
                        rot_pos = (rot_norm + 1.0) * 0.5  # 0..1

                        with effects_lock:
                            if up == [1,0,0,0,0]:
                                current_effects["gain_db"] = rot_norm * 20.0
                            elif up[:2] == [1,1] and sum(up[2:]) == 0:
                                current_effects["bitcrush_amt"] = rot_pos  # 0..1
                            elif up[:3] == [1,1,1] and sum(up[3:]) == 0:
                                # Map 0..1 to threshold 1.0..0.1
                                current_effects["clip_thresh"] = float(np.interp(rot_pos, [0,1], [1.0, 0.1]))
                            elif up[:4] == [1,1,1,1] and up[4] == 0:
                                current_effects["drive_db"] = rot_pos * 24.0
                            elif up == [1,1,1,1,1]:
                                current_effects["invert_mix"] = rot_pos

                # HUD
                cv2.rectangle(image, (0,0), (760, 150), (0,0,0), -1)
                cv2.putText(image, f"Right Hand: {up}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(image, f"Rotation: {rotation_deg_display:.1f}\u00b0", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if SOUNDDEVICE_AVAILABLE:
                    if audio_running.is_set() and not audio_error.is_set():
                        audio_status = "Audio: Active (SoundDevice)"; color = (0,255,0)
                    else:
                        audio_status = "Audio: Error"; color = (0,0,255)
                else:
                    audio_status = "Audio: Visualization Only"; color = (0,255,255)
                cv2.putText(image, audio_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                with effects_lock:
                    hud = (f"Gain:{current_effects['gain_db']:.1f}dB | "
                           f"Crush:{current_effects['bitcrush_amt']:.2f} | "
                           f"Clip:{current_effects['clip_thresh']:.2f} | "
                           f"Drive:{current_effects['drive_db']:.1f}dB | "
                           f"Invert:{current_effects['invert_mix']:.2f}")
                cv2.putText(image, hud, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (87, 255, 216), 1)

                cv2.putText(image, "Press 'q' to quit, 'r' to reset rotation", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Audio-Visual Synthesizer V3 - Custom FX", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    rotation_sums[:] = [0.0, 0.0]
                    prev_vectors[:] = [None, None]
                    print("Rotation reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        audio_running.clear()
        time.sleep(0.2)
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()