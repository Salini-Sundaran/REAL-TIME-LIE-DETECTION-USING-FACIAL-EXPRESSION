import cv2
import dlib
from scipy.spatial import distance as dist

# Load pre-trained face and landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Load video or webcam feed
cap = cv2.VideoCapture(0)

# Initialize variables
blink_counter = 0
ear_threshold = 0.2  # Adjust according to your setup
frame_counter = 0
blink_status = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        # Extract left eye and right eye coordinates
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate eye aspect ratio for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Detect eye blink
        if ear < ear_threshold:
            if not blink_status:
                blink_status = True
                blink_counter += 1
        else:
            blink_status = False

        # Display eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display eye blink count
    cv2.putText(frame, "Blink Count: {}".format(blink_counter), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

# Print total blinking count
print("Total Blinking Count:", blink_counter)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
