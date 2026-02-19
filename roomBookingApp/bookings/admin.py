from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import Room, UserProfile, Booking, Access, Record

# Register Room model
@admin.register(Room)
class RoomAdmin(admin.ModelAdmin):
    list_display = ['room_id', 'location', 'capacity', 'room_type', 'is_active', 'occupancy_display']
    list_filter = ['room_type', 'is_active']
    search_fields = ['location']
    list_editable = ['is_active']

# Register Booking model
@admin.register(Booking)
class BookingAdmin(admin.ModelAdmin):
    list_display = ['booking_id', 'user', 'room', 'booking_datetime', 'status']
    list_filter = ['status', 'booking_datetime']
    search_fields = ['user__name', 'room__location']
    date_hierarchy = 'booking_datetime'

# Register Access model
@admin.register(Access)
class AccessAdmin(admin.ModelAdmin):
    list_display = ['access_id', 'user', 'room', 'result', 'access_datetime']
    list_filter = ['result']

# Register Record model
@admin.register(Record)
class RecordAdmin(admin.ModelAdmin):
    list_display = ['record_number', 'action', 'timestamp', 'booking']
    list_filter = ['action']

# Handle UserProfile with User
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False

class CustomUserAdmin(UserAdmin):
    inlines = [UserProfileInline]
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff']

# Unregister default User and register custom
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)