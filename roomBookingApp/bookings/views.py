from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, LogoutView, PasswordResetView, PasswordResetConfirmView, PasswordChangeView
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.views.generic import TemplateView, CreateView, UpdateView, ListView, DetailView, View
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth import logout, login
from django.urls import reverse_lazy, reverse
from django.contrib.auth.models import User
from django.http import StreamingHttpResponse
from django.utils import timezone
from .models import Booking
import io
from django.shortcuts import get_object_or_404
from .models import Room, Booking
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from .models import Booking, Record
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from .models import Booking
from .forms import BookingForm, BookingEditForm
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from .models import Room
from .forms import RoomForm
from django.utils import timezone
from datetime import timedelta
from .models import Room, Booking, Access
import sys
import time
from django.utils import timezone
from datetime import timedelta
from .models import Room, Booking, Access
import cv2
from django.contrib import messages
from django.contrib.auth.models import User
from django.views import View
import numpy as np
from .forms import CustomUserCreationForm
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Face_Recognition.model_state import build_dataset_fingerprint, dataset_requires_retraining, save_training_state


# Camera streaming setup
camera_instance = None

# Face detection and recognition setup
face_cascade = None
recognizer = None
label_map = {}
face_detection_enabled = True

# Multi-frame recognition tracking
confirmed_name = None
frame_count = 0
REQUIRED_CONSISTENT_FRAMES = 5
MIN_CONFIDENCE_THRESHOLD = 40
MAX_CONFIDENCE_THRESHOLD = 120
CONFIDENCE_THRESHOLD = 75
last_prediction_distance = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_RECOGNITION_DIR = PROJECT_ROOT / "Face_Recognition"
FACE_DATASET_DIR = FACE_RECOGNITION_DIR / "dataset"
FACE_MODEL_PATH = FACE_RECOGNITION_DIR / "face_model.yml"
FACE_MODEL_STATE_PATH = FACE_RECOGNITION_DIR / "face_model_state.json"
LABEL_MAP_PATHS = [
    PROJECT_ROOT / "label_map.json",
    FACE_RECOGNITION_DIR / "label_map.json",
]


def _resolve_identity_name(raw_identity):
    identity = str(raw_identity)
    if identity.isdigit():
        user = User.objects.filter(id=int(identity)).first()
        if user is not None:
            full_name = user.get_full_name().strip()
            return full_name or user.username or f"User {identity}"
        return f"User {identity}"
    return identity


def _match_user_by_identity(identity_name):
    if not identity_name:
        return None

    value = str(identity_name).strip()
    if not value or value == "Unknown":
        return None

    if value.isdigit():
        matched_user = User.objects.filter(id=int(value), is_active=True).first()
        if matched_user:
            return matched_user

    if value.lower().startswith("user "):
        user_id = value[5:].strip()
        if user_id.isdigit():
            matched_user = User.objects.filter(id=int(user_id), is_active=True).first()
            if matched_user:
                return matched_user

    matched_user = User.objects.filter(username__iexact=value, is_active=True).first()
    if matched_user:
        return matched_user

    matched_user = User.objects.filter(email__iexact=value, is_active=True).first()
    if matched_user:
        return matched_user

    name_parts = value.split()
    if len(name_parts) >= 2:
        matched_user = User.objects.filter(
            first_name__iexact=name_parts[0],
            last_name__iexact=" ".join(name_parts[1:]),
            is_active=True,
        ).first()
        if matched_user:
            return matched_user

    for candidate in User.objects.filter(is_active=True):
        if candidate.get_full_name().strip().lower() == value.lower():
            return candidate

    return None


def _clamp_confidence_threshold(value):
    return max(MIN_CONFIDENCE_THRESHOLD, min(MAX_CONFIDENCE_THRESHOLD, int(value)))


def _load_label_map_from_disk():
    for label_map_path in LABEL_MAP_PATHS:
        if not label_map_path.exists():
            continue

        try:
            with open(label_map_path, "r", encoding="utf-8") as file:
                raw_label_map = json.load(file)

            loaded_map = {}
            for key, value in raw_label_map.items():
                loaded_map[int(key)] = _resolve_identity_name(value)

            if loaded_map:
                return loaded_map
        except Exception as error:
            print(f"Could not load label map from {label_map_path}: {error}")

    return {}


def _write_label_map_to_disk(current_label_map):
    serialized_map = {str(key): value for key, value in current_label_map.items()}
    for label_map_path in LABEL_MAP_PATHS:
        try:
            label_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_map_path, "w", encoding="utf-8") as file:
                json.dump(serialized_map, file)
        except Exception as error:
            print(f"Could not write label map to {label_map_path}: {error}")


def train_face_model_from_dataset():
    global recognizer, label_map

    if not FACE_DATASET_DIR.exists():
        recognizer = None
        label_map = {}
        return False, "No enrolled faces found. Enroll at least one user first."

    person_dirs = sorted([path for path in FACE_DATASET_DIR.iterdir() if path.is_dir()], key=lambda path: path.name)
    if not person_dirs:
        recognizer = None
        label_map = {}
        return False, "No enrolled faces found. Enroll at least one user first."

    face_images = []
    face_labels = []
    trained_label_map = {}
    label_index = 0

    for person_dir in person_dirs:
        image_paths = sorted(
            [
                *person_dir.glob("*.jpg"),
                *person_dir.glob("*.jpeg"),
                *person_dir.glob("*.png"),
            ],
            key=lambda path: path.name,
        )

        images_added = 0
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                continue

            face_images.append(image)
            face_labels.append(label_index)
            images_added += 1

        if images_added > 0:
            trained_label_map[label_index] = _resolve_identity_name(person_dir.name)
            label_index += 1

    if not face_images:
        recognizer = None
        label_map = {}
        return False, "No valid face images found in dataset folders."

    try:
        trained_recognizer = cv2.face.LBPHFaceRecognizer_create()
        trained_recognizer.train(face_images, np.array(face_labels))
        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        trained_recognizer.write(str(FACE_MODEL_PATH))

        recognizer = trained_recognizer
        label_map = trained_label_map
        _write_label_map_to_disk(label_map)
        save_training_state(FACE_MODEL_STATE_PATH, build_dataset_fingerprint(FACE_DATASET_DIR))

        return True, f"Model ready with {len(label_map)} enrolled user(s)."
    except Exception as error:
        recognizer = None
        label_map = {}
        return False, f"Could not train face model: {error}"


def _face_dataset_needs_refresh():
    """Return True when new face data has been added since the last train."""
    if not FACE_DATASET_DIR.exists():
        return False

    return dataset_requires_retraining(FACE_DATASET_DIR, FACE_MODEL_STATE_PATH)


def init_face_detection():
    """Initialize face detection and recognition models"""
    global face_cascade, recognizer, label_map
    
    try:
        # Always initialize face cascade if not already done
        if face_cascade is None:
            # Try multiple paths for Haar Cascade
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
            
            for cascade_path in cascade_paths:
                if Path(cascade_path).exists():
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not face_cascade.empty():
                        print(f"Face cascade loaded from: {cascade_path}")
                        break
            
            if face_cascade is None or face_cascade.empty():
                print("ERROR: Could not load face cascade!")
                return False
        
        # Retrain automatically when new face data was added.
        if _face_dataset_needs_refresh():
            print("New face data detected. Retraining the face model...")
            success, message = train_face_model_from_dataset()
            print(message)
            if not success:
                recognizer = None
                return False

        # Try to load recognizer model (optional - not needed for enrollment)
        if recognizer is None:
            if FACE_MODEL_PATH.exists():
                try:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(str(FACE_MODEL_PATH))
                    label_map = _load_label_map_from_disk()

                    if not label_map:
                        success, message = train_face_model_from_dataset()
                        if success:
                            print(message)
                        else:
                            print(message)
                    else:
                        print(f"Face recognition model loaded from: {FACE_MODEL_PATH}")
                except Exception as error:
                    print(f"Could not load saved face model: {error}")
                    recognizer = None

            if recognizer is None:
                success, message = train_face_model_from_dataset()
                print(message)
                if not success:
                    recognizer = None
        
        return True
    except Exception as e:
        print(f"Face detection initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_camera():
    """Get or create a singleton camera instance"""
    global camera_instance
    if camera_instance is None:
        try:
            from picamera2 import Picamera2
            print("Initializing Raspberry Pi camera...")
            camera_instance = Picamera2()
            config = camera_instance.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            camera_instance.configure(config)
            camera_instance.start()
            time.sleep(2)  # Let camera warm up
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            camera_instance = None
    return camera_instance


def generate_frames():
    """Generator function to yield camera frames with face detection as JPEG"""
    global confirmed_name, frame_count, face_detection_enabled, last_prediction_distance
    
    camera = get_camera()
    if camera is None:
        yield b'--frame\r\n'
        yield b'Content-Type: text/plain\r\n\r\n'
        yield b'Camera not available\r\n'
        return
    
    # Initialize face detection
    init_face_detection()
    
    try:
        from PIL import Image
        
        while True:
            frame = camera.capture_array()
            
            # Perform face detection and recognition if enabled and models are loaded
            if face_detection_enabled and face_cascade is not None and recognizer is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(
                    small_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Scale back up
                    x *= 2
                    y *= 2
                    w *= 2
                    h *= 2
                    
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    try:
                        # Predict face
                        label, confidence = recognizer.predict(face_roi)
                        last_prediction_distance = float(confidence)
                        
                        if confidence < CONFIDENCE_THRESHOLD:
                            current_name = label_map.get(label, "Unknown")
                        else:
                            current_name = "Unknown"
                        
                        # Multi-frame verification
                        if current_name == confirmed_name:
                            frame_count += 1
                        else:
                            confirmed_name = current_name
                            frame_count = 1
                        
                        # Determine display status
                        if frame_count >= REQUIRED_CONSISTENT_FRAMES:
                            status_name = "CONFIRMED: " + confirmed_name
                        else:
                            status_name = f"Verifying: {current_name} ({frame_count}/{REQUIRED_CONSISTENT_FRAMES})"
                        
                        # Set color based on recognition status
                        if current_name == "Unknown":
                            color = (0, 0, 255)  # Red
                        elif frame_count >= REQUIRED_CONSISTENT_FRAMES:
                            color = (0, 255, 0)  # Green
                        else:
                            color = (0, 255, 255)  # Yellow
                        
                        # Draw rectangle and text on frame
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            frame_bgr,
                            status_name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                        cv2.putText(
                            frame_bgr,
                            f"Confidence: {confidence:.1f}",
                            (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )
                    except Exception as e:
                        print(f"Face recognition error: {e}")

                if len(faces) == 0:
                    last_prediction_distance = None
                
                # Convert back to RGB for PIL
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to JPEG
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=80)
            frame_bytes = buffer.getvalue()
            
            # Yield frame in multipart format
            yield b'--frame\r\n'
            yield b'Content-Type: image/jpeg\r\n\r\n'
            yield frame_bytes
            yield b'\r\n'
            
            time.sleep(0.03)  # ~30 FPS
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Frame generation error: {e}")


def video_stream(request):
    """View to stream video frames"""
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def get_face_status(request):
    """API endpoint to get current face detection status"""
    from django.http import JsonResponse

    match_score = 0
    if last_prediction_distance is not None:
        score = 100 - (float(last_prediction_distance) / float(MAX_CONFIDENCE_THRESHOLD)) * 100
        match_score = max(0, min(100, score))
    
    return JsonResponse({
        'recognition_enabled': face_detection_enabled,
        'current_face': confirmed_name or 'None',
        'confidence': frame_count / REQUIRED_CONSISTENT_FRAMES * 100 if REQUIRED_CONSISTENT_FRAMES > 0 else 0,
        'frames_confirmed': frame_count >= REQUIRED_CONSISTENT_FRAMES,
        'match_score': match_score,
        'distance': last_prediction_distance,
        'threshold': CONFIDENCE_THRESHOLD,
        'threshold_min': MIN_CONFIDENCE_THRESHOLD,
        'threshold_max': MAX_CONFIDENCE_THRESHOLD,
    })


def set_confidence_threshold(request):
    """Set LBPH confidence threshold (higher = more lenient recognition)."""
    from django.http import JsonResponse

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required'}, status=405)

    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except Exception:
        payload = {}

    raw_value = payload.get('threshold')
    if raw_value is None:
        raw_value = request.POST.get('threshold')

    if raw_value is None:
        return JsonResponse({'success': False, 'error': 'threshold is required'}, status=400)

    try:
        threshold = _clamp_confidence_threshold(float(raw_value))
    except Exception:
        return JsonResponse({'success': False, 'error': 'threshold must be a number'}, status=400)

    global CONFIDENCE_THRESHOLD, confirmed_name, frame_count, last_prediction_distance
    CONFIDENCE_THRESHOLD = threshold
    confirmed_name = None
    frame_count = 0
    last_prediction_distance = None

    return JsonResponse({
        'success': True,
        'threshold': CONFIDENCE_THRESHOLD,
        'message': f'Recognition sensitivity updated to {CONFIDENCE_THRESHOLD}',
    })


def start_face_verification(request):
    """Prepare face model and reset status before starting verification"""
    from django.http import JsonResponse

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required'}, status=405)

    global confirmed_name, frame_count, last_prediction_distance
    confirmed_name = None
    frame_count = 0
    last_prediction_distance = None

    init_face_detection()
    success, message = train_face_model_from_dataset()
    if not success:
        return JsonResponse({'success': False, 'error': message}, status=400)

    return JsonResponse({'success': True, 'message': message})


def complete_face_login(request):
    """Create a Django login session after successful face verification."""
    from django.http import JsonResponse

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required'}, status=405)

    global confirmed_name, frame_count, last_prediction_distance

    is_verified = (
        confirmed_name is not None
        and confirmed_name != "Unknown"
        and frame_count >= REQUIRED_CONSISTENT_FRAMES
    )
    if not is_verified:
        return JsonResponse({'success': False, 'error': 'Face verification not completed'}, status=403)

    matched_user = _match_user_by_identity(confirmed_name)
    if matched_user is None:
        return JsonResponse({'success': False, 'error': f'No active user matches {confirmed_name}'}, status=404)

    login(request, matched_user, backend='django.contrib.auth.backends.ModelBackend')
    request.session['face_login_verified'] = True

    confirmed_name = None
    frame_count = 0
    last_prediction_distance = None

    return JsonResponse({
        'success': True,
        'user': matched_user.get_full_name().strip() or matched_user.username,
        'redirect_url': reverse('home'),
    })


def toggle_face_detection(request):
    """API endpoint to toggle face detection on/off"""
    from django.http import JsonResponse
    
    global face_detection_enabled
    face_detection_enabled = not face_detection_enabled
    
    return JsonResponse({
        'status': 'enabled' if face_detection_enabled else 'disabled',
        'recognition_enabled': face_detection_enabled
    })


def generate_enrollment_frames():
    """Generator function for enrollment with face detection only"""
    camera = get_camera()
    if camera is None:
        yield b'--frame\r\n'
        yield b'Content-Type: text/plain\r\n\r\n'
        yield b'Camera not available\r\n'
        return
    
    # Initialize face detection
    init_face_detection()
    
    try:
        from PIL import Image
        
        while True:
            frame = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect faces for enrollment
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                
                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        "Face Detected - Press Capture",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                
                # Convert back to RGB
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to JPEG
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            frame_bytes = buffer.getvalue()
            
            yield b'--frame\r\n'
            yield b'Content-Type: image/jpeg\r\n\r\n'
            yield frame_bytes
            yield b'\r\n'
            
            time.sleep(0.03)
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Enrollment frame generation error: {e}")


def enrollment_stream(request):
    """Stream for face enrollment"""
    return StreamingHttpResponse(
        generate_enrollment_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def capture_face_image(request):
    """Capture a single face image for enrollment"""
    from django.http import JsonResponse
    import base64
    
    camera = get_camera()
    if camera is None:
        return JsonResponse({'success': False, 'error': 'Camera not available'})
    
    init_face_detection()
    
    try:
        # Capture frame
        frame = camera.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )
            
            if len(faces) == 1:
                # Extract face
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                
                # Save to dataset directory
                user_id = request.user.id if request.user.is_authenticated else 'guest'
                dataset_path = FACE_DATASET_DIR / str(user_id)
                dataset_path.mkdir(parents=True, exist_ok=True)
                
                # Count existing images
                existing = len(list(dataset_path.glob("*.jpg")))
                filename = dataset_path / f"{existing + 1}.jpg"
                
                # Save face image
                cv2.imwrite(str(filename), face_img)
                
                return JsonResponse({
                    'success': True,
                    'count': existing + 1,
                    'message': f'Face image {existing + 1} captured'
                })
            elif len(faces) == 0:
                return JsonResponse({'success': False, 'error': 'No face detected'})
            else:
                return JsonResponse({'success': False, 'error': 'Multiple faces detected. Please ensure only one person is visible.'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Unknown error'})


def generate_verification_frames():
    """Generator function for face verification stream with real-time recognition"""
    global confirmed_name, frame_count, last_prediction_distance
    
    camera = get_camera()
    if camera is None:
        yield b'--frame\r\n'
        yield b'Content-Type: text/plain\r\n\r\n'
        yield b'Camera not available\r\n'
        return
    
    init_face_detection()
    
    try:
        from PIL import Image
        
        while True:
            frame = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect and recognize faces
            if face_cascade is not None and recognizer is not None:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    try:
                        label, confidence = recognizer.predict(face_roi)
                        last_prediction_distance = float(confidence)
                        
                        if confidence < CONFIDENCE_THRESHOLD:
                            current_name = label_map.get(label, "Unknown")
                        else:
                            current_name = "Unknown"
                        
                        # Multi-frame verification
                        if current_name == confirmed_name:
                            frame_count += 1
                        else:
                            confirmed_name = current_name
                            frame_count = 1
                        
                        # Set color based on verification progress
                        if current_name == "Unknown":
                            color = (0, 0, 255)
                        elif frame_count >= REQUIRED_CONSISTENT_FRAMES:
                            color = (0, 255, 0)
                        else:
                            color = (0, 255, 255)
                        
                        # Draw face oval/circle
                        center = (x + w//2, y + h//2)
                        axes = (w//2, int(h//1.8))
                        cv2.ellipse(frame_bgr, center, axes, 0, 0, 360, color, 3)
                        
                        # Draw status
                        if frame_count >= REQUIRED_CONSISTENT_FRAMES:
                            status = "VERIFIED"
                        else:
                            status = f"Verifying... {frame_count}/{REQUIRED_CONSISTENT_FRAMES}"
                        
                        cv2.putText(
                            frame_bgr,
                            status,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                    except Exception as e:
                        print(f"Verification error: {e}")

                if len(faces) == 0:
                    last_prediction_distance = None
                
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to JPEG
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            frame_bytes = buffer.getvalue()
            
            yield b'--frame\r\n'
            yield b'Content-Type: image/jpeg\r\n\r\n'
            yield frame_bytes
            yield b'\r\n'
            
            time.sleep(0.03)
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Verification frame generation error: {e}")


def verification_stream(request):
    """Stream for face verification"""
    return StreamingHttpResponse(
        generate_verification_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


class AdminRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser

class DebugLoginRequiredMixin(LoginRequiredMixin):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated or request.session.get('debug_no_login', False):
            return super(LoginRequiredMixin, self).dispatch(request, *args, **kwargs)
        else:
            return self.handle_no_permission()

class DebugAdminRequiredMixin(AdminRequiredMixin):
    def dispatch(self, request, *args, **kwargs):
        if request.session.get('debug_no_login', False) or self.test_func():
            return super(UserPassesTestMixin, self).dispatch(request, *args, **kwargs)
        else:
            return self.handle_no_permission()

class CustomLoginView(LoginView):
    template_name = 'auth/login.html'

class CustomRegisterView(View):
    template_name = 'auth/register.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        name = request.POST.get('name')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        role = request.POST.get('role')

        # Basic validation
        if password1 != password2:
            messages.error(request, "Passwords do not match")
            return redirect('register')

        if User.objects.filter(username=email).exists():
            messages.error(request, "Email already in use")
            return redirect('register')

        # Split name
        first_name = name.split(' ')[0]
        last_name = ' '.join(name.split(' ')[1:]) if len(name.split()) > 1 else ''

        # Create user
        user = User.objects.create_user(
            username=email,  # using email as username
            email=email,
            password=password1,
            first_name=first_name,
            last_name=last_name
        )

        # Handle role (optional)
        if role == 'admin':
            user.is_staff = True
            user.save()

        login(request, user)

        return redirect('home')

class CustomPasswordResetView(PasswordResetView):
    template_name = 'auth/password_reset.html'

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'auth/password_reset_confirm.html'

class CustomPasswordChangeView(DebugLoginRequiredMixin, PasswordChangeView):
    template_name = 'auth/password_change.html'
    form_class = PasswordChangeForm
    success_url = reverse_lazy('home')

def custom_logout(request):
    logout(request)
    return redirect('login')

@login_required
@require_POST
def cancel_booking(request, booking_id):
    try:
        profile = request.user.userprofile
        booking = Booking.objects.get(
            booking_id=booking_id,
            user=profile,
            status='CONFIRMED'
        )

        if not booking.can_cancel:
            return JsonResponse({'success': False, 'error': 'This booking cannot be cancelled anymore.'}, status=400)

        booking.status = 'CANCELLED'
        booking.save()

        # Create audit record
        Record.objects.create(
            booking=booking,
            action='BOOKING_CANCELLED',
            description=f"Booking cancelled for {booking.room.location}"
        )

        return JsonResponse({'success': True, 'message': 'Booking cancelled successfully.'})

    except Booking.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Booking not found or you cannot cancel it.'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

class HomeView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'dashboard/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile
        organisation = profile.organisation

        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        week_end = today_start + timedelta(days=7)

        # Real stats
        context['bookings_today'] = Booking.objects.filter(
            user=profile,
            booking_datetime__gte=today_start,
            booking_datetime__lt=today_end,
            status='CONFIRMED'
        ).count()

        context['rooms_available'] = Room.objects.filter(
            organisation=organisation,
            is_active=True
        ).count()   # You can make this smarter later using .is_available

        context['pending_invites'] = 0  # We'll implement invitations later if needed

        # Real data for tables
        context['today_bookings'] = Booking.objects.filter(
            user=profile,
            booking_datetime__gte=today_start,
            booking_datetime__lt=today_end,
            status='CONFIRMED'
        ).select_related('room')[:5]

        context['upcoming_bookings'] = Booking.objects.filter(
            user=profile,
            booking_datetime__gte=timezone.now(),
            status='CONFIRMED'
        ).order_by('booking_datetime')[:6]

        context['managed_rooms'] = Room.objects.filter(
            organisation=organisation,
            is_active=True
        )[:5]

        # Rooms user can access = all active rooms in organisation for now
        context['accessible_rooms'] = Room.objects.filter(
            organisation=organisation,
            is_active=True
        )[:5]

        # Recent access attempts (real data)
        context['recent_access'] = Access.objects.filter(
            room__organisation=organisation
        ).select_related('room', 'user')[:5]

        return context

class LiveFeedView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'live_system/live_feed.html'

class UserProfileView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'user_management/user_profile.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile
        context['profile'] = profile
        context['bookings_count'] = profile.bookings.count()
        return context

class UserEditView(DebugLoginRequiredMixin, UpdateView):
    model = User
    fields = ['first_name', 'last_name', 'email'] 
    template_name = 'user_management/user_edit.html'
    success_url = reverse_lazy('user_profile')

    def get_object(self):
        return self.request.user

class UserNotificationsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'user_management/user_notifications.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        profile = self.request.user.userprofile

        context['notifications'] = profile.notifications.all().order_by('-created_at')

        return context

class UserManagementView(DebugAdminRequiredMixin, ListView):
    model = User
    template_name = 'user_management/user_management.html'
    context_object_name = 'users'

from .models import Room

class RoomListView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_list.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile

        rooms = Room.objects.filter(
            organisation=profile.organisation
        ).order_by('location')

        context['rooms'] = rooms
        context['total_rooms'] = rooms.count()

        return context

class RoomOverviewView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_overview.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile

        room_id = self.request.GET.get('room_id')

        try:
            if room_id:
                room = get_object_or_404(Room, room_id=room_id, organisation=profile.organisation)
            else:
                # Fallback: show first active room
                room = Room.objects.filter(
                    organisation=profile.organisation, 
                    is_active=True
                ).first()
        except:
            room = None

        if not room:
            context['room'] = None
            context['error'] = "Room not found or you don't have access to it."
            return context

        context['room'] = room

        # Real upcoming bookings for this specific room
        context['upcoming_bookings'] = Booking.objects.filter(
            room=room,
            booking_datetime__gte=timezone.now(),
            status='CONFIRMED'
        ).select_related('user__user').order_by('booking_datetime')[:6]

        # Recent access attempts for this room
        context['recent_access'] = Access.objects.filter(
            room=room
        ).select_related('user__user').order_by('-access_datetime')[:5]

        return context

class RoomCreationView(DebugLoginRequiredMixin, CreateView):
    model = Room
    form_class = RoomForm
    template_name = 'room_management/room_creation.html'
    success_url = reverse_lazy('room_list')

    def form_valid(self, form):
        # Automatically assign the room to the user's organisation
        form.instance.organisation = self.request.user.userprofile.organisation
        messages.success(self.request, f'Room "{form.instance.location}" created successfully!')
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, 'Please correct the errors below.')
        return super().form_invalid(form)

class RoomEditView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_edit.html'

class RoomPermissionsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_permissions.html'

class RoomLogView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_log.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile

        context['records'] = Record.objects.filter(
            room__organisation=profile.organisation
        ).select_related('booking', 'room', 'user').order_by('-timestamp')[:50]

        return context

class BookingCreationView(DebugLoginRequiredMixin, CreateView):
    model = Booking
    form_class = BookingForm
    template_name = 'booking_system/booking_creation.html'
    success_url = reverse_lazy('my_bookings')   # or 'room_list' if you prefer

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.instance.user = self.request.user.userprofile
        messages.success(self.request, f'Booking for {form.instance.room.location} created successfully!')
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, 'Please fix the errors below.')
        return super().form_invalid(form)

    def get_initial(self):
        initial = super().get_initial()
        room_id = self.request.GET.get('room_id')
        if room_id:
            try:
                initial['room'] = int(room_id)
            except:
                pass
        return initial

class BookingEditView(DebugLoginRequiredMixin, UpdateView):
    model = Booking
    form_class = BookingEditForm
    template_name = 'booking_system/booking_edit.html'
    success_url = reverse_lazy('my_bookings')

    def get_queryset(self):
        return Booking.objects.filter(user=self.request.user.userprofile)

    def form_valid(self, form):
        messages.success(self.request, 'Booking updated successfully!')
        return super().form_valid(form)

from .models import Booking

class MyBookingsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/my_bookings.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = self.request.user.userprofile

        # Upcoming: Confirmed bookings in the future
        context['upcoming_bookings'] = Booking.objects.filter(
            user=profile,
            booking_datetime__gte=timezone.now(),
            status='CONFIRMED'
        ).select_related('room').order_by('booking_datetime')

        # Past: All bookings that are either in the past OR cancelled/completed
        context['past_bookings'] = Booking.objects.filter(
            user=profile
        ).exclude(
            booking_datetime__gte=timezone.now(),
            status='CONFIRMED'
        ).select_related('room').order_by('-booking_datetime')

        context['has_upcoming'] = context['upcoming_bookings'].exists()
        context['has_past'] = context['past_bookings'].exists()

        return context

class BookingInvitationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_invitation.html'

class FaceEnrollmentView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/face_enrollment.html'

class FaceVerificationView(TemplateView):
    template_name = 'face_recognition/face_verification.html'

class AccessResultView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/access_result.html'

class SystemStatusView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'live_system/system_status.html'

class AdminGlobalAuditLogView(DebugAdminRequiredMixin, TemplateView):
    template_name = 'admin/admin_global_audit_log.html'

class AdminSettingsView(DebugAdminRequiredMixin, TemplateView):
    template_name = 'admin/admin_settings.html'

class ReportsView(DebugAdminRequiredMixin, TemplateView):
    template_name = 'admin/reports.html'

class Error403View(TemplateView):
    template_name = 'error_legal/403.html'

class Error404View(TemplateView):
    template_name = 'error_legal/404.html'

class Error500View(TemplateView):
    template_name = 'error_legal/500.html'

class PrivacyBiometricConsentView(TemplateView):
    template_name = 'error_legal/privacy_biometric_consent.html'

# DEBUG TOGGLE
class ToggleDebugView(View):
    def get(self, request, *args, **kwargs):
        request.session['debug_no_login'] = not request.session.get('debug_no_login', False)
        return redirect(request.GET.get('next', 'home'))