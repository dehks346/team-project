#!/usr/bin/env python3
"""
Quick test script to verify camera and face detection setup
"""

import cv2
import time
from pathlib import Path

print("=" * 60)
print("Testing Camera and Face Detection Setup")
print("=" * 60)

# Test 1: Check OpenCV installation
print("\n1. Testing OpenCV installation...")
print(f"   OpenCV version: {cv2.__version__}")

# Test 2: Check Haar Cascade availability
print("\n2. Testing Haar Cascade availability...")
cascade_paths = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]

# Try cv2.data if it exists
try:
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade_paths.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    pass

cascade_found = None
for path in cascade_paths:
    if Path(path).exists():
        print(f"   ✓ Found cascade at: {path}")
        cascade_found = path
        break

if cascade_found:
    face_cascade = cv2.CascadeClassifier(cascade_found)
    if face_cascade.empty():
        print("   ✗ ERROR: Cascade loaded but is empty!")
    else:
        print("   ✓ Cascade loaded successfully")
else:
    print("   ✗ ERROR: No Haar Cascade file found!")

# Test 3: Test Picamera2
print("\n3. Testing Raspberry Pi Camera...")
try:
    from picamera2 import Picamera2
    print("   ✓ Picamera2 module imported successfully")
    
    print("   Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print("   ✓ Camera started successfully")
    
    time.sleep(1)
    
    print("   Capturing test frame...")
    frame = picam2.capture_array()
    print(f"   ✓ Captured frame: {frame.shape}")
    
    picam2.stop()
    print("   ✓ Camera stopped successfully")
    
except ImportError as e:
    print(f"   ✗ ERROR: Cannot import Picamera2: {e}")
except Exception as e:
    print(f"   ✗ ERROR: Camera test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check face recognition model
print("\n4. Checking face recognition model...")
model_path = Path(__file__).parent / "Face_Recognition" / "face_model.yml"
if model_path.exists():
    print(f"   ✓ Model found at: {model_path}")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model_path))
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Model exists but failed to load: {e}")
else:
    print(f"   ⚠ Model not found at: {model_path}")
    print("   (This is OK for enrollment, but needed for verification)")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
