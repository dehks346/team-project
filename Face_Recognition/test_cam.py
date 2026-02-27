import time
import cv2
import numpy as np
from pathlib import Path


# ========================
# LOAD FACE RECOGNITION MODEL
# ========================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Face_Recognition/face_model.yml")

# Label map for recognized faces
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


def show_with_picamera2():
	from picamera2 import Picamera2

	picam2 = Picamera2()
	config = picam2.create_preview_configuration(main={"size": (640, 480)})
	picam2.configure(config)
	picam2.start()
	time.sleep(0.5)

	cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
	while True:
		frame = picam2.capture_array()
		
		# Convert BGR to grayscale for face detection
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Detect faces
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)
		
		# Process each detected face
		for (x, y, w, h) in faces:
			# Draw rectangle around face
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
			
			# Extract face ROI for recognition
			face_roi = gray[y:y+h, x:x+w]
			
			# Recognize face
			try:
				label_id, confidence = recognizer.predict(face_roi)
				
				# Display label and confidence if confidence is good
				if confidence < 100:  # Lower confidence = better match
					label = label_map.get(label_id, "Unknown")
					text = f"{label} ({confidence:.0f})"
					color = (0, 255, 0)  # Green for recognized
				else:
					text = f"Unknown ({confidence:.0f})"
					color = (0, 0, 255)  # Red for unknown
				
				cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
							0.8, color, 2)
			except:
				cv2.putText(frame, "No Model", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
							0.8, (0, 0, 255), 2)
		
		cv2.imshow("Face Recognition", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	picam2.stop()
	cv2.destroyAllWindows()


def show_with_opencv():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open camera at /dev/video0")

	cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
	while True:
		ret, frame = cap.read()
		if not ret:
			continue

		cv2.imshow("Camera", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


def main():
	try:
		show_with_picamera2()
	except Exception:
		show_with_opencv()


if __name__ == "__main__":
	main()
