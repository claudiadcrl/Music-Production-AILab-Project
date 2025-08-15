import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Gain, Delay, PitchShift, Distortion
import threading
import queue
import time

'''
Redesigned audio engine:
- Audio callback is lightweight: it only pulls preprocessed blocks from a queue.
- A DSP worker thread reads current controls and applies ONLY lightweight effects in real-time
  (Gain, Distortion, Reverb, Delay).
- Pitch shifting is handled by a SEPARATE background worker that pre-renders a pitch-shifted
  copy of the entire loop when the desired semitone value changes significantly. The DSP worker
  mixes that pre-rendered path in at runtime.
This avoids real-time PitchShift in the callback and prevents underflows/glitches.
'''

# ===================== Shared control state (Vision -> Audio) =====================

# Globals shared between threads (vision updates these values)
current_effects = {
    "gain":        0.0,   # dB
    "pitch":       0.0,   # desired semitones (-12..+12). Used to choose which pitch loop to mix
    "distortion":  0.0,   # drive dB
    "reverb":      0.25,  # room_size (0..1)
    "delay":       0.25   # delay_seconds (0..1 typically)
}
effects_lock = threading.Lock()  # protects current_effects

# ============================== Audio / DSP state ================================

# Load audio file (stereo loop expected)
AUDIO_FILE = 'C:/Users/compu/Downloads/Alesis-Fusion-Bass-Loop.wav'
data, samplerate = sf.read(AUDIO_FILE, always_2d=True)
channels = data.shape[1]
loop_len = len(data)

# Choose a LARGE blocksize for stability (you can lower later if CPU allows)
blocksize = 16384  # ~371 ms at 44.1kHz
audio_queue = queue.Queue(maxsize=200)  # deep buffer for robustness

# Loop position (owned by DSP worker)
loop_position = 0

# Pitch path state
pitch_state_lock = threading.Lock()
pitch_current_semitones = None        # semitones for which pitch_data is valid
pitch_target_semitones = 0.0          # last requested semitones from controls
pitch_data = None                     # np.ndarray same shape as data (pre-rendered pitch loop)

# Crossfade management when pitch buffer updates
pitch_prev_data = None                # previous pitch buffer for crossfade
pitch_crossfade_blocks_remaining = 0  # blocks to crossfade over


# ============================== Utility Functions ================================

def get_loop_chunk(frames: int):
    #Return a contiguous chunk from the dry loop, wrapping seamlessly.
    global loop_position
    end_pos = loop_position + frames
    if end_pos <= loop_len:
        chunk = data[loop_position:end_pos]
        loop_position = 0 if end_pos == loop_len else end_pos
        return chunk
    # Wrap
    part1 = data[loop_position:]
    part2 = data[:end_pos - loop_len]
    loop_position = end_pos - loop_len
    return np.vstack((part1, part2))


def get_pitch_chunk(frames: int, start_pos: int):
    #Return a chunk from the current pitch_data, aligned to dry loop position
    global pitch_data
    if pitch_data is None:
        return None
    end_pos = start_pos + frames
    if end_pos <= loop_len:
        return pitch_data[start_pos:end_pos]
    part1 = pitch_data[start_pos:]
    part2 = pitch_data[:end_pos - loop_len]
    return np.vstack((part1, part2))


def db_to_linear(db):
    return 10.0 ** (db / 20.0)


# ============================== Pitch Worker =====================================

def regenerate_pitch_buffer(target_semitones: float):
    #Pre-render a full-loop pitch-shifted buffer for the current file
    global pitch_data
    # Apply pitch offline to the WHOLE loop. This is heavy but happens outside RT audio.
    # Note: Using a fresh Pedalboard with PitchShift only.
    board = Pedalboard([PitchShift(semitones=float(target_semitones))])
    # Process in reasonably sized blocks to avoid huge temporary buffers
    block = 65536
    out = []
    for start in range(0, loop_len, block):
        end = min(loop_len, start + block)
        out.append(board(data[start:end], samplerate))
    pitch_data = np.vstack(out).astype(np.float32, copy=False)


def pitch_worker(stop_event: threading.Event):
    global pitch_current_semitones, pitch_target_semitones
    global pitch_prev_data, pitch_crossfade_blocks_remaining

    last_requested = None
    while not stop_event.is_set():
        # Snapshot desired semitones from controls
        with effects_lock:
            desired = float(current_effects.get("pitch", 0.0))
        # Quantize to the nearest half-semitone to avoid thrashing
        desired_q = round(desired * 2.0) / 2.0

        # If desired pitch changed significantly, regenerate buffer
        if (last_requested is None) or (abs(desired_q - last_requested) >= 0.5):
            last_requested = desired_q
            # Remember for DSP mix
            with pitch_state_lock:
                pitch_target_semitones = desired_q
            # Do heavy work: regenerate buffer
            if abs(desired_q) < 0.01:
                # Zero pitch means no shift -> reset to None to save CPU in DSP path
                with pitch_state_lock:
                    pitch_prev = None
                    pitch_prev_data = None
                    pitch_current_semitones = 0.0
                    # Signal that pitch_data is not used
                    globals()['pitch_data'] = None
            else:
                # Keep a copy for crossfade
                with pitch_state_lock:
                    old_buf = globals()['pitch_data']
                try:
                    regenerate_pitch_buffer(desired_q)
                    with pitch_state_lock:
                        # Prepare crossfade from old to new over N blocks
                        if old_buf is not None:
                            globals()['pitch_prev_data'] = old_buf
                            globals()['pitch_crossfade_blocks_remaining'] = 8
                        else:
                            globals()['pitch_prev_data'] = None
                            globals()['pitch_crossfade_blocks_remaining'] = 0
                        globals()['pitch_current_semitones'] = desired_q
                except Exception as e:
                    # If pitch rendering fails, disable pitch path for safety
                    with pitch_state_lock:
                        globals()['pitch_data'] = None
                        globals()['pitch_prev_data'] = None
                        globals()['pitch_crossfade_blocks_remaining'] = 0
                        globals()['pitch_current_semitones'] = None

        # Sleep a bit before checking again
        stop_event.wait(0.2)


# ============================== DSP Worker =======================================

def apply_lightweight_effects(x: np.ndarray, effects_snapshot: dict):
    '''"Apply cheap-ish effects in real-time.\n
    We avoid PitchShift here.\n
    effects_snapshot fields: gain (dB), distortion (dB drive), reverb (room_size), delay (seconds)\n
    '''
    chain = []

    # Gain
    gain_db = float(effects_snapshot.get("gain", 0.0))
    if abs(gain_db) > 0.01:
        chain.append(Gain(gain_db=gain_db))

    # Distortion
    drive_db = float(effects_snapshot.get("distortion", 0.0))
    if drive_db > 0.01:
        chain.append(Distortion(drive_db=drive_db))

    # Reverb
    room_size = float(effects_snapshot.get("reverb", 0.0))
    if room_size > 0.01:
        chain.append(Reverb(room_size=room_size))

    # Delay
    delay_seconds = float(effects_snapshot.get("delay", 0.0))
    if delay_seconds > 0.01:
        chain.append(Delay(delay_seconds=delay_seconds, feedback=0.3, mix=0.25))

    if not chain:
        return x

    board = Pedalboard(chain)
    return board(x, samplerate)


def dsp_worker(stop_event: threading.Event):
    global pitch_crossfade_blocks_remaining

    pos_for_pitch = 0  # keep a local pointer to align pitch chunk with dry chunk
    while not stop_event.is_set():
        # 1) Read current controls (cheap)
        with effects_lock:
            snapshot = current_effects.copy()

        # 2) Pull dry chunk (no effects)
        start_pos = pos_for_pitch
        dry_chunk = get_loop_chunk(blocksize)
        pos_for_pitch = (start_pos + blocksize) % loop_len

        # 3) Optional pitch path (pre-rendered) — get aligned chunk
        with pitch_state_lock:
            pd = globals()['pitch_data']
            prev_pd = globals()['pitch_prev_data']
            xfade_blocks = globals()['pitch_crossfade_blocks_remaining']
            target_semi = globals()['pitch_target_semitones']

        pitched = None
        if pd is not None:
            pitched = get_pitch_chunk(blocksize, start_pos)

            # Crossfade from previous buffer if we just switched
            if prev_pd is not None and xfade_blocks > 0:
                prev_chunk = None
                # construct prev chunk aligned
                end_pos = start_pos + blocksize
                if end_pos <= loop_len:
                    prev_chunk = prev_pd[start_pos:end_pos]
                else:
                    part1 = prev_pd[start_pos:]
                    part2 = prev_pd[:end_pos - loop_len]
                    prev_chunk = np.vstack((part1, part2))
                alpha = (xfade_blocks - 1) / max(1, xfade_blocks)
                pitched = alpha * prev_chunk + (1.0 - alpha) * pitched
                with pitch_state_lock:
                    globals()['pitch_crossfade_blocks_remaining'] -= 1
                    if globals()['pitch_crossfade_blocks_remaining'] <= 0:
                        globals()['pitch_prev_data'] = None

        # 4) Mix pitch into dry according to a wet amount derived from magnitude
        #    If you prefer a separate pitch mix param, add it to current_effects and use here.
        if pitched is not None:
            wet = min(1.0, abs(float(snapshot.get("pitch", 0.0))) / 12.0)  # 0..1
            out = (1.0 - wet) * dry_chunk + wet * pitched
        else:
            out = dry_chunk

        # 5) Apply lightweight real-time effects
        processed = apply_lightweight_effects(out, snapshot).astype(np.float32, copy=False)

        # 6) Enqueue for audio callback
        try:
            audio_queue.put(processed, timeout=0.5)
        except queue.Full:
            # Drop the oldest and try again to keep latency bounded
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_queue.put(processed, timeout=0.1)
            except queue.Full:
                pass


# ============================== Audio Callback ===================================

def audio_callback(outdata, frames, time_info, status):
    if status:
        print(status)
    try:
        block = audio_queue.get_nowait()
    except queue.Empty:
        block = np.zeros((frames, channels), dtype=np.float32)
    if block.shape[0] != frames:
        out = np.zeros((frames, channels), dtype=np.float32)
        n = min(frames, block.shape[0])
        out[:n] = block[:n]
        outdata[:] = out
    else:
        outdata[:] = block


def start_audio(stop_event: threading.Event):
    # Start workers
    t_dsp = threading.Thread(target=dsp_worker, args=(stop_event,), daemon=True)
    t_pitch = threading.Thread(target=pitch_worker, args=(stop_event,), daemon=True)
    t_dsp.start()
    t_pitch.start()

    # Stream
    with sd.OutputStream(channels=channels,
                         callback=audio_callback,
                         blocksize=blocksize,
                         samplerate=samplerate):
        # Keep thread alive until stop
        while not stop_event.is_set():
            time.sleep(0.1)


# ============================== Visual (unchanged) ===============================

joint_list = [[4,2,1], [8,6,5], [12,10,9], [16,14,13], [20,18,17]]
up=[0,0,0,0,0]

def get_finger_names(lista):
    s=""
    names=["Thumb","Index","Middle","Ring","Pinky"]
    for i,el in enumerate(lista):
        if el!=0:
            s+=names[i]+" "
    return s

def draw_finger_angles(image, landmrk, label, joint_list):
    #Loop through joint sets 
    count = 0
    for i,joint in enumerate(joint_list):
        a = np.array([landmrk[joint[0]].x, landmrk[joint[0]].y]) # First coord
        b = np.array([landmrk[joint[1]].x, landmrk[joint[1]].y]) # Second coord
        c = np.array([landmrk[joint[2]].x, landmrk[joint[2]].y]) # Third coord
        
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if angle > 180.0:
            angle = 360-angle
        if label=="Right":
            if angle<150:
                up[i]=0
            else:
                up[i]=1
        if label=="Left":
            if angle>150:
                count+=1

    cv2.putText(image, f"Fingers {label}: {count if label=="Left" else up}", (30, 80 if label=="Right" else 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return image

def get_angle(v1,v2):
    angle1=np.arctan2(v1[1],v1[0])
    angle2=np.arctan2(v2[1],v2[0])
    return angle2-angle1

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            output = text, coords
    return output

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap=cv2.VideoCapture(0) #0 is laptop, 1 is IVCam (IPad)
prev_vectors = [None, None]
rotation_sums = [0.0, 0.0]

# Start audio thread
stop_event = threading.Event()
audio_thread = threading.Thread(target=start_audio, args=(stop_event,), daemon=True)
audio_thread.start()

# Vision loop
with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(216, 255, 87), thickness=2, circle_radius=2),
                )
                num=hand_handedness.classification[0].index
                if len(results.multi_hand_landmarks)>1:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    anchor = landmarks[0][:2] #Wrist
                    index_tip = landmarks[8][:2]  #Index finger tip
                    curr_vector = index_tip - anchor
                    
                    if len(prev_vectors)>=num+1:
                        if prev_vectors[num] is not None:
                            angle = get_angle(prev_vectors[num], curr_vector)
                            rotation_sums[num] += angle
                            total_deg = np.degrees(rotation_sums[1])
                            if num==1:
                                cv2.putText(image, f"Rotation Total: {total_deg:.2f}", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                        prev_vectors[num] = curr_vector
                    if get_label(num, hand_landmarks, results):
                        text, coord = get_label(num, hand_landmarks, results)
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                    label = hand_handedness.classification[0].label
                    draw_finger_angles(image, hand_landmarks.landmark, label, joint_list)
    
        rot_deg=np.degrees(rotation_sums[1])
        rot_norm = np.clip(rot_deg / 180, -1, 1)  # normalize to [-1,1]

        with effects_lock:
            if up == [1,0,0,0,0]:
                current_effects["gain"] = rot_norm * 20   # ±20dB
            elif up[:2] == [1,1] and up[2:] == [0,0,0]:
                # pitch controls: semitones desired; magnitude also acts as wet amount in mix
                current_effects["pitch"] = rot_norm * 12  # ±12 semitones
            elif up[:3] == [1,1,1] and up[3:] == [0,0]:
                current_effects["distortion"] = max(0, rot_norm) * 15  # 0-15dB drive
            elif up[:4] == [1,1,1,1] and up[4] == 0:
                current_effects["reverb"] = float(np.interp(rot_norm, [-1,1], [0.1,0.9]))
            elif up == [1,1,1,1,1]:
                current_effects["delay"] = float(np.interp(rot_norm, [-1,1], [0.05,0.75]))
        
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stop_event.set()
time.sleep(0.2)