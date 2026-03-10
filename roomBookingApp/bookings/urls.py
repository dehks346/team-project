from django.urls import path
from .views import (
    CustomLoginView, CustomRegisterView, CustomPasswordResetView, CustomPasswordResetConfirmView,
    CustomPasswordChangeView, custom_logout, HomeView, UserProfileView, UserEditView,
    UserNotificationsView, UserManagementView, RoomListView, RoomOverviewView, RoomCreationView,
    RoomEditView, RoomPermissionsView, RoomLogView, BookingCreationView, BookingEditView,
    MyBookingsView, BookingInvitationView, FaceEnrollmentView, FaceVerificationView,
    AccessResultView, LiveFeedView, SystemStatusView, AdminGlobalAuditLogView,
    AdminSettingsView, ReportsView, Error403View, Error404View, Error500View,
    PrivacyBiometricConsentView, ToggleDebugView, video_stream, get_face_status, toggle_face_detection,
    enrollment_stream, capture_face_image, verification_stream, start_face_verification
)

urlpatterns = [
    path('login/', CustomLoginView.as_view(), name='login'),
    path('register/', CustomRegisterView.as_view(), name='register'),
    path('password_reset/', CustomPasswordResetView.as_view(), name='password_reset'),
    path('password_reset_confirm/<uidb64>/<token>/', CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password_change/', CustomPasswordChangeView.as_view(), name='password_change'),
    path('logout/', custom_logout, name='logout'),

    path('', HomeView.as_view(), name='home'),

    path('user/profile/', UserProfileView.as_view(), name='user_profile'),
    path('user/edit/', UserEditView.as_view(), name='user_edit'),
    path('user/notifications/', UserNotificationsView.as_view(), name='user_notifications'),
    path('user/management/', UserManagementView.as_view(), name='user_management'),

    path('rooms/', RoomListView.as_view(), name='room_list'),
    path('rooms/overview/', RoomOverviewView.as_view(), name='room_overview'),
    path('rooms/create/', RoomCreationView.as_view(), name='room_creation'),
    path('rooms/edit/', RoomEditView.as_view(), name='room_edit'),
    path('rooms/permissions/', RoomPermissionsView.as_view(), name='room_permissions'),
    path('rooms/log/', RoomLogView.as_view(), name='room_log'),

    path('bookings/create/', BookingCreationView.as_view(), name='booking_creation'),
    path('bookings/edit/', BookingEditView.as_view(), name='booking_edit'),
    path('bookings/my/', MyBookingsView.as_view(), name='my_bookings'),
    path('bookings/invitation/', BookingInvitationView.as_view(), name='booking_invitation'),

    path('face/enrollment/', FaceEnrollmentView.as_view(), name='face_enrollment'),
    path('face/enrollment_stream/', enrollment_stream, name='enrollment_stream'),
    path('face/capture_image/', capture_face_image, name='capture_face_image'),
    path('face/verification/', FaceVerificationView.as_view(), name='face_verification'),
    path('face/start_verification/', start_face_verification, name='start_face_verification'),
    path('face/verification_stream/', verification_stream, name='verification_stream'),
    path('face/access_result/', AccessResultView.as_view(), name='access_result'),

    path('live/feed/', LiveFeedView.as_view(), name='live_feed'),
    path('live/video_stream/', video_stream, name='video_stream'),
    path('live/face_status/', get_face_status, name='face_status'),
    path('live/toggle_face_detection/', toggle_face_detection, name='toggle_face_detection'),
    path('live/status/', SystemStatusView.as_view(), name='system_status'),

    path('admin/audit_log/', AdminGlobalAuditLogView.as_view(), name='admin_global_audit_log'),
    path('admin/settings/', AdminSettingsView.as_view(), name='admin_settings'),
    path('admin/reports/', ReportsView.as_view(), name='reports'),

    path('403/', Error403View.as_view(), name='error_403'),
    path('404/', Error404View.as_view(), name='error_404'),
    path('500/', Error500View.as_view(), name='error_500'),
    path('privacy_biometric_consent/', PrivacyBiometricConsentView.as_view(), name='privacy_biometric_consent'),

    path('toggle_debug/', ToggleDebugView.as_view(), name='toggle_debug'),
]