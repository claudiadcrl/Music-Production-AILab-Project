#I'm just testing some code from ChatGPT to have a knob gesture
import cv2
import mediapipe as mp
import numpy as np

def get_angle(v1, v2):
    """Calculate angle between two 2D vectors in radians"""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    return angle2 - angle1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_vector = None
rotation_sum = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            anchor = landmarks[0][:2] #Wrist
            index_tip = landmarks[8][:2]  #Index finger tip

            curr_vector = index_tip - anchor
            if prev_vector is not None:
                angle = get_angle(prev_vector, curr_vector)
                rotation_sum += angle

                # Optional: degrees
                angle_deg = np.degrees(angle)
                total_deg = np.degrees(rotation_sum)

                cv2.putText(image, f"Δ Angle: {angle_deg:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Total: {total_deg:.2f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            prev_vector = curr_vector

    cv2.imshow("Knob Rotation Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()