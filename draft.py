import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Gain, Compressor, Bitcrush, Distortion
import threading

''' Threads work both
    GAIN WORKS
    BITCRUSH WORKS
    DISTORSION WORKS
    the others don't work, doesn't change anything'''

#two threads: a visual thread (to track hands) and an audio thread (to control audio)

# globals shared between threads
current_effects = {
    "gain":        0.0,
    "bitcrush":     8,
    "distortion":  0.0,
    "reverb":      0.5,
    "compressor":       0.0
}

effects_lock = threading.Lock() #so that writes from the vision side and reads from the audio side never collide

#------------------------------------------------------------------------------------------ AUDIO -----------------------------------------------------------------------------------------------------------
# Load audio file
AUDIO_FILE = 'C:/Users/compu/Downloads/Alesis-Fusion-Bass-Loop.wav'
data, samplerate = sf.read(AUDIO_FILE, always_2d=True)
loop_position = 0
blocksize = 1024 #23 ms latency

def audio_callback(outdata, frames, time, status):
    global loop_position

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
    
    #to apply effects we need the dictionary with the values of the effects given by hand input
    #dictionary is global and shared by the two theads, but we operate on copy to avoid modifications
    #must be used inside lock to avoid problems

    with effects_lock:
        effects=current_effects.copy() #gets current state of parameters changed by input
    
    #make the board (we need to make a new one if parameters change)
    board = Pedalboard([
        Bitcrush(bit_depth=effects["bitcrush"]),
        Distortion(drive_db=effects["distortion"]),
        Gain(gain_db=effects["gain"]),
        Compressor(threshold_db=effects["compressor"], ratio=8),
        Reverb(room_size=effects["reverb"], wet_level=0.7, dry_level=0.1)
    ])

    outdata[:] = board(chunk, samplerate)

#now start the audio stream, but it need to be in the thread, so we make a function
def start_audio():
    with sd.OutputStream(channels=2, callback=audio_callback,
                     blocksize=blocksize, samplerate=samplerate):
        threading.Event().wait()  # sleep forever

#make thread and start it
audio_thread = threading.Thread(target=start_audio, daemon=True) #daemon --> runs in background
audio_thread.start()



#----------------------------------------------------------------------------------------- VISUAL -----------------------------------------------------------------------------------------------------------

joint_list = [[4,2,1], [8,6,5], [12,10,9], [16,14,13], [20,18,17]]
up=[0,0,0,0,0]

#FUNCTIONS
def get_finger_names(lista):
    str=""
    names=["Thumb","Index","Middle","Ring","Pinky"]
    for i,el in enumerate(lista):
        if el!=0:
            str+=names[i]+" "
    return str

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
                if angle<160:
                    up[i]=0
                else:
                    up[i]=1
            if label=="Left":
                if angle>150:
                    count+=1

    cv2.putText(image, f"Fingers {label}: {count if label=="Left" else up}", (30, 80 if label=="Right" else 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return image

#Function to calculate the angle between two 2D vectors in radians
def get_angle(v1,v2):
    angle1=np.arctan2(v1[1],v1[0])
    angle2=np.arctan2(v2[1],v2[0])
    return angle2-angle1

#Function to get labels of hands and coords
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
    return output

#initialise hands and drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#gets camera and initialises vector and final rotation angle
cap=cv2.VideoCapture(0) #0 is laptop, 1 is IVCam (IPad)
prev_vectors = [None, None]
rotation_sums = [0.0, 0.0]

#track only two hands
with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        #get frame
        ret, frame = cap.read() #ret is the return value, indicates if successful
        if not ret:
            break

        #BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Flip on horizontal
        image = cv2.flip(image, 1)
        
        #Set flag
        image.flags.writeable = False
        
        #Detections
        results = hands.process(image)
        
        #Set flag to true
        image.flags.writeable = True
        
        #RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Detections
        #print(results)

        #Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                #draw the landmarks on the hand
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(216, 255, 87), thickness=2, circle_radius=2),
                )
                num=hand_handedness.classification[0].index
                if len(results.multi_hand_landmarks)>1:
                    #Set anchor point and compute rotation angle
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    anchor = landmarks[0][:2] #Wrist
                    index_tip = landmarks[8][:2]  #Index finger tip
                    curr_vector = index_tip - anchor
                    
                    if len(prev_vectors)>=num+1:
                        if prev_vectors[num] is not None:
                            angle = get_angle(prev_vectors[num], curr_vector)
                            rotation_sums[num] += angle

                            #angle_deg = np.degrees(angle)
                            total_deg = np.degrees(rotation_sums[1])

                            # Display hand-specific data
                            '''cv2.putText(image, f"Hand {num+1} Δ: {angle_deg:.2f}", (30, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)'''
                            if num==1:
                                cv2.putText(image, f"Rotation Total: {total_deg:.2f}", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                        prev_vectors[num] = curr_vector
                    # Render left or right detection
                    if get_label(num, hand_landmarks, results):
                        text, coord = get_label(num, hand_landmarks, results)
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                    # Draw angles to image from joint list
                    label = hand_handedness.classification[0].label
                    draw_finger_angles(image, hand_landmarks.landmark, label, joint_list)

            sel = sum(up)
            rot_deg=np.degrees(rotation_sums[1])
            rot_clamped = np.clip(rot_deg, -60, 60) # Clamp rotation to -50 - +50 degrees
            rot_norm = rot_clamped / 60.0  # normalizes so that it's in between -1 and 1 (and we can just add or subtract)

            with effects_lock: #inside lock to avoid reads and/or other modifications
                if rot_deg<180:
                    if up == [1,0,0,0,0]:
                        current_effects["gain"] = rot_norm * 20   # ±20dB

                    elif up == [1,1,0,0,0]:
                        current_effects["bitcrush"] = ((rot_norm + 1) / 2) * 6 + 2  # in range [4,12] bit

                    elif up== [1,1,1,0,0]:
                        current_effects["distortion"] = (max(0, rot_norm) * 55)  # 0-55dB drive

                    elif up== [1,1,1,1,0]:
                        current_effects["reverb"] = (rot_norm + 1) / 2

                    elif up == [1,1,1,1,1]:
                        current_effects["compressor"] = ((rot_norm + 1) / 2) * (-50)
            
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()