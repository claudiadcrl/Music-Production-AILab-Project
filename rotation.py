'''Get the rotation of the hands using the wrists landmarks (landmark 0) as an anchor point

- It computes the current vector that goes from wrist to the point of the index and stores the previous one
- Computes the angle between the previous angle and the x axis (using arctan2 function)
and the angle between current angle and x axis
- Takes the difference between the two angles to know how much it has moved in a frame
- Adds it to total sum of angles to get rotation

Used UI from Youtube video tutorial (the jupyter notebook one), but I'll change it later'''

'''Given the list of joints, it also computes the angles of the fingers to understand when they're closed or open,
according to a threshold (I've set it at 150 degrees):

-In the joint list, we have the landmarks for each finger with tip, middle point and base of the finger
-The function iterates through the joint list
- At each step, it computes the angle between the tip and the x axis at middle point(using arctan2 function)
and the angle between base and x axis at middle point
- Takes the difference between the two angles
- Check which hand it is: if it's the right hand, updates the list; if it's the left hand, updates the counter

For the right hand it knows which exact fingers are up (so to avoid any "illegal" movements when changing parameters),
while for the left hand we just need to know how many fingers are up to select the track
'''

import cv2
import mediapipe as mp
import numpy as np

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
    
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



