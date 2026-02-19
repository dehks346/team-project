import cv2
import numpy as np
import time



# ========================
# LOAD FACE RECOGNITION MODEL
# ========================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# 🔹 PASTE LABEL MAP HERE
label_map = {
    0: "Nell",
    1: "charli",
    2: "elliot",
    3: "henry",
    4: "jed",
    5: "olly",
    6: "stan"
}

# ========================
# LOAD FACE DETECTOR
# ========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ========================
# CAMERA SETUP
# ========================
cap = cv2.VideoCapture(0)

print("Starting face_detect")

# === LOAD OPENCV FACE DETECTOR ===
# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Face detector loaded")

# === CAMERA SETUP ===
# Change 0 to 1 if you have multiple webcams
cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cv2.namedWindow("Face Detect", cv2.WINDOW_NORMAL)
print("Window created")




# === MULTI FRAME RECOGNITION SETUP ===
# To improve accuracy, we can keep track of predictions over multiple frames
# and only show a name if it's consistent for a few frames.
confirmed_name = None
frame_count = 0


REQUIRED_CONSISTENT_FRAMES = 5
CONFIDENCE_THRESHOLD = 50  # Lower = more strict (try 40-60 for better unknown detection)






# === FPS CALCULATION ===
fps_counter = 0
fps_start_time = time.time()
current_fps = 0




# === MAIN LOOP ===

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # (Optional) small frame performance boost
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Scale back up
        x *= 2
        y *= 2
        w *= 2
        h *= 2

        # 🔹 Extract the face region
        face_roi = gray[y:y+h, x:x+w]

        # 🔹 THIS IS WHERE YOUR LINES GO
        label, confidence = recognizer.predict(face_roi)
        
        # Debug: print confidence to help tune threshold
        print(f"Label: {label_map.get(label, 'Unknown')}, Confidence: {confidence:.1f}")

        if confidence < CONFIDENCE_THRESHOLD:
            current_name = label_map.get(label, "Unknown")
        else:
            current_name = "Unknown"
        # Check if the current prediction is consistent with the last few frames
        if current_name == confirmed_name:
            frame_count += 1
        else:
            confirmed_name = current_name
            frame_count = 1

        if frame_count >= REQUIRED_CONSISTENT_FRAMES:
            status_name = "CONFIRMED: " + confirmed_name
        else:
            status_name = f"Verifying: {current_name} ({frame_count}/{REQUIRED_CONSISTENT_FRAMES})"

        # Set color based on status
        if current_name == "Unknown":
            color = (0, 0, 255)  # Red for unknown faces
        elif frame_count >= REQUIRED_CONSISTENT_FRAMES:
            color = (0, 255, 0)  # Green for confirmed
        else:
            color = (0, 255, 255)  # Yellow for verifying

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            status_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        # Show confidence level below the rectangle
        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Face Detect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Clean exit")
