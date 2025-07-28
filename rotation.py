'''Get the rotation of the hands using the wrists landmarks (landmark 0) as an anchor point

- It computes the current vector that goes from wrist to the point of the index and stores the previous one
- Computes the angle between the previous angle and the x axis (using arctan2 function)
and the angle between current angle and x axis
- Takes the difference between the two angles to know how much it has moved in a frame
- Adds it to total sum of angles to get rotation

Used UI from Youtube video tutorial (the jupyter notebook one), but I'll change it later'''

#There might be some imbalance with the anchor point, but first figure out the audio


import cv2
import mediapipe as mp
import numpy as np

#FUNCTIONS
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
            
            # Extract Coordinat.es
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
    return output

#initialise hands and drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#gets camera and initialises vector and final rotation angle
cap=cv2.VideoCapture(0)
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
        print(results)

        #Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                #draw the landmarks on the hand
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(216, 255, 87), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(195, 255, 0), thickness=2, circle_radius=2),
                )
                
                #Set anchor point and compute rotation angle
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
                anchor = landmarks[0][:2] #Wrist
                index_tip = landmarks[8][:2]  #Index finger tip
                curr_vector = index_tip - anchor

                if prev_vectors[num] is not None:
                    angle = get_angle(prev_vectors[num], curr_vector)
                    rotation_sums[num] += angle

                    angle_deg = np.degrees(angle)
                    total_deg = np.degrees(rotation_sums[num])

                    # Display hand-specific data
                    y_offset = 50 + num * 30
                    '''cv2.putText(image, f"Hand {num+1} Δ: {angle_deg:.2f}", (30, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)'''
                    cv2.putText(image, f"Hand {num+1} Total: {total_deg:.2f}", (30, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)

                prev_vectors[num] = curr_vector
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



