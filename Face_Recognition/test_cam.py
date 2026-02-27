import time
from picamera2 import Picamera2, Preview


def main():
	"""
	Simple camera test using pure picamera2 without OpenCV.
	Uses built-in preview window to display camera feed.
	Press Ctrl+C to stop.
	"""
	print("Initializing picamera2...")
	
	picam2 = Picamera2()
	
	# Create preview configuration (let picamera2 choose the best format)
	preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
	
	picam2.configure(preview_config)
	
	# Start with built-in preview window
	# Try QTGL first, fallback to DRM if that fails
	print("Starting camera preview...")
	print("Press Ctrl+C to stop")
	
	try:
		picam2.start_preview(Preview.QTGL)
	except Exception as e:
		print(f"QTGL preview failed ({e}), trying DRM preview...")
		try:
			picam2.start_preview(Preview.DRM)
		except Exception as e2:
			print(f"DRM preview failed ({e2}), running without preview...")
	
	picam2.start()
	
	try:
		# Keep running until interrupted
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		print("\nStopping camera...")
	finally:
		picam2.stop()
		print("Camera stopped.")


if __name__ == "__main__":
	main()
