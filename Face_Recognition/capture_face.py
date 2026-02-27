import cv2
import os
import time
from pathlib import Path
from picamera2 import Picamera2

name = "elliot"  # CHANGE THIS
save_path = f"dataset/{name}"

os.makedirs(save_path, exist_ok=True)

# ========================
# LOAD FACE DETECTOR
# ========================
def resolve_haar_cascade_path():
	data_path = getattr(cv2, "data", None)
	if data_path and getattr(data_path, "haarcascades", None):
		return str(Path(data_path.haarcascades) / "haarcascade_frontalface_default.xml")
	
	candidates = [
		"/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
		"/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
	]
	for candidate in candidates:
		if Path(candidate).exists():
			return candidate
	
	return None

cascade_path = resolve_haar_cascade_path()
if not cascade_path:
	raise RuntimeError(
		"Could not locate haarcascade_frontalface_default.xml. "
		"Install OpenCV data files or set a valid path."
	)

face_cascade = cv2.CascadeClassifier(cascade_path)

# ========================
# SETUP PICAMERA2
# ========================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(0.5)

count = 0
face_img = None

print("Look at the camera. Press SPACE to capture. Q to quit.")

while True:
	frame = picam2.capture_array()
	
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

picam2.stop()
cv2.destroyAllWindows()
print("Done")
