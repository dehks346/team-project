import cv2
import os
import time

# ========================
# CONFIG
# ========================
DATASET_DIR = "dataset"

LABELS = {
    ord('j'): "jed",
    ord('a'): "alice"
}

MAX_IMAGES_PER_PERSON = 50

# ========================
# SETUP
# ========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("Press key to label face:")
for k, v in LABELS.items():
    print(f"  {chr(k)} → {v}")
print("Press Q to quit")

counts = {name: 0 for name in LABELS.values()}

# ========================
# MAIN LOOP
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Draw all detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Live Face Labeling", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Only save if EXACTLY ONE face
    if key in LABELS and len(faces) == 1:
        name = LABELS[key]

        save_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]

        count = len(os.listdir(save_dir)) + 1
        if count <= MAX_IMAGES_PER_PERSON:
            cv2.imwrite(f"{save_dir}/{count}.jpg", face_img)
            print(f"Saved {name}: {count}")
        else:
            print(f"{name} already has enough images")

    elif key in LABELS:
        print("Make sure exactly ONE face is visible")

# ========================
# CLEANUP
# ========================
cap.release()
cv2.destroyAllWindows()
print("Done")
