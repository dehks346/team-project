import cv2
import os
import time

name = "elliot"  # CHANGE THIS
save_path = f"dataset/{name}"

os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

print("Look at the camera. Press SPACE to capture. Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = gray[y:y+h, x:x+w]

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE
        if len(faces) == 1:
            count += 1
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            print(f"Saved image {count}")
        else:
            print("Make sure exactly ONE face is visible")

    if key == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print("Done")
