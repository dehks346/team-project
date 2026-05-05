from django.shortcuts import render, redirect
from django import forms
from django.contrib.auth.views import LoginView, LogoutView, PasswordResetView, PasswordResetConfirmView, PasswordChangeView
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.forms import AuthenticationForm
from django.views.generic import TemplateView, CreateView, UpdateView, ListView, DetailView, View
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib import messages
from django.contrib.auth import logout, login
from django.urls import reverse_lazy, reverse
from django.contrib.auth.models import User
from django.db import models
from django.http import StreamingHttpResponse
from urllib.parse import urlencode
import io
import json
import sys
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from pathlib import Path
from django.utils.dateparse import parse_date, parse_time
from django.utils import timezone
from django.contrib.auth.models import User
from django.http import JsonResponse
from .models import Room, UserProfile, Booking, BookingInvitation, Access, Organisation
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
DASHBOARD_REDIRECT_URL_NAME = 'home'
FACE_BOOKING_PENDING_ROOM_SESSION_KEY = 'face_booking_pending_room_id'
FACE_BOOKING_REDIRECT_SESSION_KEY = 'face_booking_redirect_url'
FACE_BOOKING_VERIFIED_ROOM_SESSION_KEY = 'face_booking_verified_room_id'


def _get_room_by_id(room_id):
    if not room_id:
        return None
    try:
        return Room.objects.filter(room_id=int(room_id), is_active=True).first()
    except (TypeError, ValueError):
        return None


def _room_requires_face_verification(room_id):
    room = _get_room_by_id(room_id)
    return bool(room and room.is_face_required)


def _booking_creation_url(room_id=None):
    booking_url = reverse('booking_creation')
    if not room_id:
        return booking_url
    return f"{booking_url}?{urlencode({'room': room_id})}"


def _face_verification_url(next_url=None, room_id=None, purpose='login'):
    params = {'purpose': purpose}
    if next_url:
        params['next'] = next_url
    if room_id:
        params['room'] = room_id
    return f"{reverse('face_verification')}?{urlencode(params)}"


def _get_or_create_user_profile(user):
    profile, _ = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            'name': user.get_full_name().strip() or user.username,
            'email': user.email or f'{user.username}@example.com',
            'phone_number': '',
        },
    )
    if not profile.email:
        profile.email = user.email or f'{user.username}@example.com'
        profile.save(update_fields=['email'])
    return profile


def _get_user_profile(user):
    if not user or not user.is_authenticated:
        return None
    return UserProfile.objects.select_related('organisation').filter(user=user).first()


def _get_user_organisation(user):
    profile = _get_user_profile(user)
    return profile.organisation if profile and profile.organisation_id else None


def _is_organisation_user(user):
    return _get_user_organisation(user) is not None


def _room_queryset_for_user(user, active_only=True):
    rooms = Room.objects.select_related('organisation').order_by('room_id')
    if active_only:
        rooms = rooms.filter(is_active=True)

    if not user or not user.is_authenticated or user.is_superuser:
        return rooms

    user_org = _get_user_organisation(user)
    if user_org is not None:
        return rooms.filter(organisation=user_org)

    return rooms.filter(organisation__isnull=True)


def _parse_equipment(raw_equipment):
    if not raw_equipment:
        return []
    if isinstance(raw_equipment, list):
        return [item for item in raw_equipment if str(item).strip()]
    return [item.strip() for item in str(raw_equipment).split(',') if item.strip()]


def _parse_invite_tokens(raw_invites):
    if not raw_invites:
        return []
    return [token.strip() for token in str(raw_invites).split(',') if token.strip()]


def _resolve_invitee(token):
    matched_user = User.objects.filter(email__iexact=token).first()
    if matched_user:
        return matched_user, matched_user.email, matched_user.get_full_name().strip() or matched_user.username

    matched_user = User.objects.filter(username__iexact=token).first()
    if matched_user:
        return matched_user, matched_user.email, matched_user.get_full_name().strip() or matched_user.username

    for candidate in User.objects.filter(is_active=True):
        if candidate.get_full_name().strip().lower() == token.lower():
            return candidate, candidate.email, candidate.get_full_name().strip() or candidate.username

    return None, token, token


class WebsiteUserCreationForm(UserCreationForm):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
    ]

    name = forms.CharField(max_length=150)
    email = forms.EmailField(required=True)
    role = forms.ChoiceField(choices=ROLE_CHOICES)
    username = forms.CharField(required=False, widget=forms.HiddenInput())
    organisation = forms.ModelChoiceField(queryset=Organisation.objects.all(), required=False)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ('name', 'email', 'role', 'username', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        self.allow_admin_creation = kwargs.pop('allow_admin_creation', False)
        super().__init__(*args, **kwargs)
        if not self.allow_admin_creation:
            self.fields['role'].choices = [('user', 'User')]

    def _generate_username_from_email(self, email):
        base_username = (email.split('@')[0] if '@' in email else email).strip().lower() or 'user'
        candidate = base_username
        suffix = 1

        while User.objects.filter(username__iexact=candidate).exists():
            suffix += 1
            candidate = f"{base_username}{suffix}"

        return candidate

    def _extract_email_domain(self, email):
        value = (email or '').strip().lower()
        if '@' not in value:
            return ''
        return value.split('@', 1)[1].strip()

    def _organisation_for_email(self, email):
        domain = self._extract_email_domain(email)
        if not domain:
            return None

        organisations = Organisation.objects.all()

        # First pass: exact domain match.
        for organisation in organisations:
            org_domain = self._extract_email_domain(organisation.email_address)
            if org_domain and org_domain == domain:
                return organisation

        # Second pass: support subdomains like dept.company.com -> company.com.
        for organisation in organisations:
            org_domain = self._extract_email_domain(organisation.email_address)
            if org_domain and domain.endswith(f".{org_domain}"):
                return organisation

        return None

    def clean_email(self):
        email = (self.cleaned_data.get('email') or '').strip().lower()
        if not email:
            raise forms.ValidationError('Email is required.')
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('An account with this email already exists.')
        return email

    def clean_role(self):
        role = self.cleaned_data.get('role')
        if role == 'admin' and not self.allow_admin_creation:
            raise forms.ValidationError('Only admins can create admin accounts.')
        return role

    def clean_username(self):
        username = (self.cleaned_data.get('username') or '').strip()
        if username:
            return username

        email = (self.data.get('email') or '').strip().lower()
        if not email:
            return username

        return self._generate_username_from_email(email)

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')

        if email:
            cleaned_data['email'] = email.strip().lower()

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        full_name = (self.cleaned_data.get('name') or '').strip()
        name_parts = full_name.split()

        user.email = self.cleaned_data['email']
        user.first_name = name_parts[0] if name_parts else ''
        user.last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''

        role = self.cleaned_data.get('role', 'user')
        user.is_staff = role == 'admin'
        user.is_superuser = role == 'admin'

        if commit:
            user.save()

        # Ensure UserProfile exists (signals may have created one) and update organisation
        profile_defaults = {
            'name': full_name,
            'email': user.email,
            'phone_number': ''
        }
        org = self.cleaned_data.get('organisation') or self._organisation_for_email(user.email)
        profile, _ = UserProfile.objects.get_or_create(user=user, defaults=profile_defaults)
        if org:
            profile.organisation = org
        # update name/email if provided
        profile.name = profile_defaults['name']
        profile.email = profile_defaults['email']
        profile.save()

        return user


class EmailOrUsernameAuthenticationForm(AuthenticationForm):
    ACCOUNT_TYPE_CHOICES = [
        ('regular', 'Regular user'),
        ('organisation', 'Organisation user'),
        ('admin', 'Admin'),
    ]

    account_type = forms.ChoiceField(choices=ACCOUNT_TYPE_CHOICES, required=False)

    def clean(self):
        username_or_email = (self.cleaned_data.get('username') or '').strip()

        if username_or_email and '@' in username_or_email:
            matched_user = User.objects.filter(email__iexact=username_or_email).first()
            if matched_user is not None:
                self.cleaned_data['username'] = matched_user.get_username()

        cleaned_data = super().clean()
        account_type = (self.data.get('account_type') or 'regular').strip().lower()
        user = self.get_user()

        if user is None:
            return cleaned_data

        if account_type == 'admin' and not user.is_superuser:
            raise forms.ValidationError('This account is not an admin account.')

        if account_type == 'organisation' and not _is_organisation_user(user):
            raise forms.ValidationError('This account is not linked to an organisation profile.')

        cleaned_data['account_type'] = account_type
        return cleaned_data

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
    matched_profile = _get_user_profile(matched_user)
    request.session['is_org_user'] = bool(matched_profile and matched_profile.organisation_id)
    request.session['organisation_name'] = matched_profile.organisation.name if matched_profile and matched_profile.organisation_id else ''

    pending_room = request.session.get(FACE_BOOKING_PENDING_ROOM_SESSION_KEY)
    booking_redirect_url = request.session.get(FACE_BOOKING_REDIRECT_SESSION_KEY)
    redirect_url = reverse(DASHBOARD_REDIRECT_URL_NAME)

    if booking_redirect_url and _room_requires_face_verification(pending_room):
        request.session[FACE_BOOKING_VERIFIED_ROOM_SESSION_KEY] = pending_room
        request.session.pop(FACE_BOOKING_PENDING_ROOM_SESSION_KEY, None)
        request.session.pop(FACE_BOOKING_REDIRECT_SESSION_KEY, None)
        redirect_url = booking_redirect_url

    confirmed_name = None
    frame_count = 0
    last_prediction_distance = None

    return JsonResponse({
        'success': True,
        'user': matched_user.get_full_name().strip() or matched_user.username,
        'redirect_url': redirect_url,
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
        if request.user.is_authenticated:
            profile = _get_user_profile(request.user)
            request.session['is_org_user'] = bool(profile and profile.organisation_id)
            request.session['organisation_name'] = profile.organisation.name if profile and profile.organisation_id else ''

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


class OrganisationOrAdminRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser or _is_organisation_user(self.request.user)


class DebugOrganisationOrAdminRequiredMixin(OrganisationOrAdminRequiredMixin):
    def dispatch(self, request, *args, **kwargs):
        if request.session.get('debug_no_login', False) or self.test_func():
            return super(UserPassesTestMixin, self).dispatch(request, *args, **kwargs)
        return self.handle_no_permission()

class CustomLoginView(LoginView):
    template_name = 'auth/login.html'
    authentication_form = EmailOrUsernameAuthenticationForm

    def form_valid(self, form):
        response = super().form_valid(form)
        profile = _get_user_profile(self.request.user)
        self.request.session['is_org_user'] = bool(profile and profile.organisation_id)
        self.request.session['organisation_name'] = profile.organisation.name if profile and profile.organisation_id else ''
        return response

    def form_invalid(self, form):
        for error in form.non_field_errors():
            messages.error(self.request, error)
        return super().form_invalid(form)

    def get_success_url(self):
        return reverse_lazy(DASHBOARD_REDIRECT_URL_NAME)

class CustomRegisterView(CreateView):
    form_class = WebsiteUserCreationForm
    template_name = 'auth/register.html'
    success_url = reverse_lazy('home')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['allow_admin_creation'] = self.request.user.is_authenticated and self.request.user.is_superuser
        return kwargs

    def form_valid(self, form):
        self.object = form.save()
        if not self.request.user.is_authenticated:
            login(self.request, self.object)

        active_user = self.request.user if self.request.user.is_authenticated else self.object
        profile = _get_user_profile(active_user)
        self.request.session['is_org_user'] = bool(profile and profile.organisation_id)
        self.request.session['organisation_name'] = profile.organisation.name if profile and profile.organisation_id else ''

        if profile and profile.organisation_id:
            messages.success(self.request, f'Account created successfully under {profile.organisation.name}.')
        else:
            messages.success(self.request, 'Account created successfully.')
        return redirect(self.get_success_url())

    def form_invalid(self, form):
        for errors in form.errors.values():
            for error in errors:
                messages.error(self.request, error)
        return super().form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['organisations'] = Organisation.objects.all().order_by('name')
        return context

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
    request.session.pop('is_org_user', None)
    request.session.pop('organisation_name', None)
    return redirect('login')

class HomeView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'dashboard/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user_profile = _get_or_create_user_profile(self.request.user)
        now = timezone.now()
        today = timezone.localdate()
        week_end = today + timedelta(days=7)

        user_bookings = Booking.objects.filter(user=user_profile).select_related('room')
        today_bookings = user_bookings.filter(booking_datetime__date=today).order_by('booking_datetime')
        upcoming_week_bookings = user_bookings.filter(
            booking_datetime__date__gt=today,
            booking_datetime__date__lte=week_end,
        ).order_by('booking_datetime')

        user_invites = BookingInvitation.objects.filter(
            models.Q(invited_user=self.request.user) | models.Q(invited_email__iexact=self.request.user.email)
        ).select_related('booking', 'booking__room')

        pending_invites = user_invites.filter(status='PENDING').order_by('-created_at')

        active_rooms = list(_room_queryset_for_user(self.request.user, active_only=True))
        available_rooms_count = sum(1 for room in active_rooms if room.is_available)

        context['bookings_today_count'] = today_bookings.count()
        context['rooms_available_count'] = available_rooms_count
        context['pending_invites_count'] = pending_invites.count()

        context['today_bookings'] = today_bookings[:5]
        context['upcoming_week_bookings'] = upcoming_week_bookings[:5]
        context['managed_rooms'] = active_rooms[:5]
        context['accessible_rooms'] = active_rooms[:5]
        context['pending_invites'] = pending_invites[:5]
        context['recent_access_attempts'] = Access.objects.select_related('room', 'user').order_by('-access_datetime')[:5]
        context['now'] = now
        return context

class LiveFeedView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'live_system/live_feed.html'

class UserProfileView(DebugLoginRequiredMixin, DetailView):
    model = User
    template_name = 'user_management/user_profile.html'
    context_object_name = 'user_obj'
    def get_object(self):
        return self.request.user

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile = _get_or_create_user_profile(self.request.user)
        context['profile'] = profile
        context['bookings_count'] = Booking.objects.filter(user=profile).count()
        context['is_organisation_user'] = bool(profile.organisation_id)
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
        user = self.request.user
        context['invitations'] = BookingInvitation.objects.filter(
            models.Q(invited_user=user) | models.Q(invited_email__iexact=user.email)
        ).order_by('-created_at')
        return context

class UserManagementView(DebugOrganisationOrAdminRequiredMixin, ListView):
    model = User
    template_name = 'user_management/user_management.html'
    context_object_name = 'users'

    def get_queryset(self):
        queryset = super().get_queryset().select_related('userprofile__organisation').order_by('first_name', 'last_name', 'username')
        if self.request.user.is_superuser:
            return queryset

        organisation = _get_user_organisation(self.request.user)
        if organisation is None:
            return queryset.none()

        return queryset.filter(userprofile__organisation=organisation)

class RoomListView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_list.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['rooms'] = _room_queryset_for_user(self.request.user, active_only=True)
        return context

class RoomOverviewView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_overview.html'

class RoomCreationView(DebugOrganisationOrAdminRequiredMixin, TemplateView):
    template_name = 'room_management/room_creation.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['organisations'] = Organisation.objects.all().order_by('name')
        context['current_organisation'] = _get_user_organisation(self.request.user)
        return context

    def post(self, request, *args, **kwargs):
        room_name = (request.POST.get('name') or '').strip()
        location = (request.POST.get('location') or '').strip()
        capacity = request.POST.get('capacity') or 0
        room_type = request.POST.get('room_type') or 'MEET'
        is_face_required = request.POST.get('is_face_required') == 'on'
        approval_required = request.POST.get('approval_required') == 'on'
        opening_time = parse_time(request.POST.get('opening_time') or '')
        closing_time = parse_time(request.POST.get('closing_time') or '')
        equipment = _parse_equipment(request.POST.get('equipment'))
        photo_name = ''

        uploaded_photo = request.FILES.get('photo')
        if uploaded_photo:
            photo_name = uploaded_photo.name

        if not room_name or not location:
            messages.error(request, 'Room name and location are required.')
            return self.get(request, *args, **kwargs)

        room_organisation = None
        if request.user.is_superuser:
            organisation_id = (request.POST.get('organisation') or '').strip()
            if organisation_id:
                room_organisation = Organisation.objects.filter(pk=organisation_id).first()
                if room_organisation is None:
                    messages.error(request, 'Selected organisation does not exist.')
                    return self.get(request, *args, **kwargs)
        else:
            room_organisation = _get_user_organisation(request.user)
            if room_organisation is None and not request.session.get('debug_no_login', False):
                messages.error(request, 'You must belong to an organisation to create rooms.')
                return redirect('room_list')

        try:
            capacity_value = int(capacity)
        except (TypeError, ValueError):
            messages.error(request, 'Capacity must be a number.')
            return self.get(request, *args, **kwargs)

        Room.objects.create(
            name=room_name,
            location=location,
            capacity=capacity_value,
            room_type=room_type,
            is_face_required=is_face_required,
            approval_required=approval_required,
            equipment=equipment,
            opening_time=opening_time,
            closing_time=closing_time,
            photo_name=photo_name,
            organisation=room_organisation,
        )

        messages.success(request, f'Room "{room_name}" created successfully.')
        return redirect('room_list')

class RoomEditView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_edit.html'

class RoomPermissionsView(DebugAdminRequiredMixin, TemplateView):
    template_name = 'room_management/room_permissions.html'

class RoomLogView(DebugAdminRequiredMixin, TemplateView):
    template_name = 'room_management/room_log.html'


class BookingFaceGateView(DebugLoginRequiredMixin, View):
    def get(self, request, room_id, *args, **kwargs):
        booking_url = _booking_creation_url(room_id)

        if not _room_requires_face_verification(room_id):
            return redirect(booking_url)

        request.session[FACE_BOOKING_PENDING_ROOM_SESSION_KEY] = str(room_id)
        request.session[FACE_BOOKING_REDIRECT_SESSION_KEY] = booking_url
        return redirect(_face_verification_url(next_url=booking_url, room_id=room_id, purpose='booking'))

class BookingCreationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_creation.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        room_id = self.request.GET.get('room')
        context['rooms'] = _room_queryset_for_user(self.request.user, active_only=True)

        selected_room = _get_room_by_id(room_id)
        if selected_room and not _room_queryset_for_user(self.request.user, active_only=True).filter(room_id=selected_room.room_id).exists():
            selected_room = None

        context['selected_room'] = selected_room
        return context

    def dispatch(self, request, *args, **kwargs):
        room_id = (request.GET.get('room') or '').strip()

        if _room_requires_face_verification(room_id):
            verified_room = request.session.get(FACE_BOOKING_VERIFIED_ROOM_SESSION_KEY)

            if verified_room != room_id:
                booking_url = _booking_creation_url(room_id)
                request.session[FACE_BOOKING_PENDING_ROOM_SESSION_KEY] = room_id
                request.session[FACE_BOOKING_REDIRECT_SESSION_KEY] = booking_url
                return redirect(_face_verification_url(next_url=booking_url, room_id=room_id, purpose='booking'))

            request.session.pop(FACE_BOOKING_VERIFIED_ROOM_SESSION_KEY, None)

        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        room_id = request.GET.get('room') or request.POST.get('room')
        room = _get_room_by_id(room_id)
        if room is None:
            messages.error(request, 'Please select a valid room.')
            return redirect('booking_creation')

        if not _room_queryset_for_user(request.user, active_only=True).filter(room_id=room.room_id).exists():
            messages.error(request, 'You do not have access to this room.')
            return redirect('booking_creation')

        user_profile = _get_or_create_user_profile(request.user)

        booking_date = parse_date(request.POST.get('booking_date') or '')
        start_time = parse_time(request.POST.get('start_time') or '')
        duration_minutes = int(request.POST.get('duration') or 60)
        notes = (request.POST.get('notes') or '').strip()
        recurring = request.POST.get('recurring') == 'on'
        recurrence_pattern = (request.POST.get('recurrence_pattern') or '').strip()
        invitees = _parse_invite_tokens(request.POST.get('invitees'))

        if not booking_date or not start_time:
            messages.error(request, 'Booking date and start time are required.')
            return redirect(f"{reverse('booking_creation')}?room={room.room_id}")

        booking_datetime = timezone.make_aware(datetime.combine(booking_date, start_time))

        booking = Booking.objects.create(
            user=user_profile,
            room=room,
            booking_datetime=booking_datetime,
            duration_minutes=duration_minutes,
            notes=notes,
            is_recurring=recurring,
            recurrence_pattern=recurrence_pattern,
        )

        for token in invitees:
            invited_user, invited_email, invited_name = _resolve_invitee(token)
            BookingInvitation.objects.create(
                booking=booking,
                invited_user=invited_user,
                invited_email=invited_email,
                invited_name=invited_name,
            )

        messages.success(request, 'Booking saved successfully.')
        return redirect('my_bookings')

class BookingEditView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_edit.html'

class MyBookingsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/my_bookings.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user_profile = UserProfile.objects.filter(user=self.request.user).first()
        if user_profile:
            all_bookings = Booking.objects.filter(user=user_profile).select_related('room').order_by('-booking_datetime')
        else:
            all_bookings = Booking.objects.none()

        now = timezone.now()
        context['bookings'] = all_bookings
        context['upcoming_bookings'] = all_bookings.filter(booking_datetime__gte=now)
        context['past_bookings'] = all_bookings.filter(booking_datetime__lt=now)
        context['invitations'] = BookingInvitation.objects.filter(
            models.Q(invited_user=self.request.user) | models.Q(invited_email__iexact=self.request.user.email)
        ).order_by('-created_at')
        return context

class BookingInvitationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_invitation.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['invitations'] = BookingInvitation.objects.filter(
            models.Q(invited_user=self.request.user) | models.Q(invited_email__iexact=self.request.user.email)
        ).order_by('-created_at')
        return context


class BookingInvitationRespondView(DebugLoginRequiredMixin, View):
    def post(self, request, invite_id, *args, **kwargs):
        invite = BookingInvitation.objects.filter(invitation_id=invite_id).first()
        if invite is None:
            messages.error(request, 'Invitation not found.')
            return redirect('booking_invitation')

        is_target_user = invite.invited_user == request.user or (invite.invited_email and invite.invited_email.lower() == request.user.email.lower())
        if not is_target_user:
            messages.error(request, 'You cannot modify this invitation.')
            return redirect('booking_invitation')

        action = (request.POST.get('action') or '').strip().lower()
        if action == 'accept':
            invite.mark_accepted()
            messages.success(request, 'Invitation accepted.')
        elif action == 'decline':
            invite.mark_declined()
            messages.success(request, 'Invitation declined.')
        else:
            messages.error(request, 'Invalid invitation action.')

        return redirect('booking_invitation')

class FaceEnrollmentView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/face_enrollment.html'


class OrganisationProfileView(DebugOrganisationOrAdminRequiredMixin, TemplateView):
    template_name = 'user_management/organisation_profile.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        organisation = _get_user_organisation(self.request.user)
        if self.request.user.is_superuser:
            organisation_id = (self.request.GET.get('organisation') or '').strip()
            if organisation_id:
                organisation = Organisation.objects.filter(pk=organisation_id).first()

        context['organisation'] = organisation
        context['all_organisations'] = Organisation.objects.all().order_by('name') if self.request.user.is_superuser else []

        if organisation is None:
            context['organisation_users'] = UserProfile.objects.none()
            context['organisation_rooms'] = Room.objects.none()
            context['organisation_bookings_count'] = 0
            return context

        organisation_users = UserProfile.objects.filter(organisation=organisation).select_related('user').order_by('name')
        organisation_rooms = Room.objects.filter(organisation=organisation).order_by('name')
        organisation_bookings_count = Booking.objects.filter(room__organisation=organisation).count()

        context['organisation_users'] = organisation_users
        context['organisation_rooms'] = organisation_rooms[:8]
        context['organisation_bookings_count'] = organisation_bookings_count
        return context

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