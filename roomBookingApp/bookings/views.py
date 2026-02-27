from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, LogoutView, PasswordResetView, PasswordResetConfirmView, PasswordChangeView
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.views.generic import TemplateView, CreateView, UpdateView, ListView, DetailView, View
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth import logout
from django.urls import reverse_lazy
from django.contrib.auth.models import User
from django.http import StreamingHttpResponse
import cv2
import threading  


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

class CustomRegisterView(CreateView):
    form_class = UserCreationForm 
    template_name = 'auth/register.html'
    success_url = reverse_lazy('home')

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

class HomeView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'dashboard/home.html'

class UserProfileView(DebugLoginRequiredMixin, DetailView):
    model = User
    template_name = 'user_management/user_profile.html'
    context_object_name = 'user_obj'
    def get_object(self):
        return self.request.user

class UserEditView(DebugLoginRequiredMixin, UpdateView):
    model = User
    fields = ['first_name', 'last_name', 'email'] 
    template_name = 'user_management/user_edit.html'
    success_url = reverse_lazy('user_profile')

    def get_object(self):
        return self.request.user

class UserNotificationsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'user_management/user_notifications.html'

class UserManagementView(DebugAdminRequiredMixin, ListView):
    model = User
    template_name = 'user_management/user_management.html'
    context_object_name = 'users'

class RoomListView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_list.html'

class RoomOverviewView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_overview.html'

class RoomCreationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_creation.html'

class RoomEditView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_edit.html'

class RoomPermissionsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_permissions.html'

class RoomLogView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'room_management/room_log.html'

class BookingCreationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_creation.html'

class BookingEditView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_edit.html'

class MyBookingsView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/my_bookings.html'

class BookingInvitationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'booking_system/booking_invitation.html'

class FaceEnrollmentView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/face_enrollment.html'

class FaceVerificationView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/face_verification.html'

class AccessResultView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'face_recognition/access_result.html'

class LiveFeedView(DebugLoginRequiredMixin, TemplateView):
    template_name = 'live_system/live_feed.html'

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


# CAMERA STREAMING
class VideoCamera:
    """
    Class to handle video camera operations with thread-safe frame access.
    """
    def __init__(self):
        # Initialize camera - 0 for default camera, or use IP camera URL
        self.video = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        
        # For Raspberry Pi camera module, you can use:
        # self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Or for IP camera/RTSP stream:
        # self.video = cv2.VideoCapture('rtsp://username:password@ip_address:port/stream')
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        """
        Capture a frame from the camera and encode it as JPEG.
        """
        with self.lock:
            success, image = self.video.read()
            if not success:
                return None
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()


def gen_frames(camera):
    """
    Generator function to yield video frames in multipart format.
    """
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


class VideoFeedView(DebugLoginRequiredMixin, View):
    """
    View to stream video frames to the client.
    """
    def get(self, request, *args, **kwargs):
        camera = VideoCamera()
        return StreamingHttpResponse(
            gen_frames(camera),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )


# DEBUG TOGGLE
class ToggleDebugView(View):
    def get(self, request, *args, **kwargs):
        request.session['debug_no_login'] = not request.session.get('debug_no_login', False)
        return redirect(request.GET.get('next', 'home'))