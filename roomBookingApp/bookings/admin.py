from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import Organisation, Room, UserProfile, Booking, Access, Record

# Register Room model
@admin.register(Room)
class RoomAdmin(admin.ModelAdmin):
    list_display = ['room_id', 'organisation', 'location', 'capacity', 'room_type', 'is_active', 'occupancy_display']
    list_filter = ['room_type', 'is_active', 'organisation']
    search_fields = ['location', 'organisation__name']
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


@admin.register(Organisation)
class OrganisationAdmin(admin.ModelAdmin):
    list_display = ['organisation_id', 'name', 'email_address', 'phone_number', 'number_of_users', 'number_of_rooms',
                    'fee']
    list_filter = ['created_at']
    search_fields = ['name', 'email_address', 'unique_access_code']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = (
        ('Organisation Info', {
            'fields': ('name', 'email_address', 'password', 'unique_access_code', 'phone_number', 'address')
        }),
        ('Billing', {
            'fields': ('fee',)
        }),
        ('Stats', {
            'fields': ('number_of_users', 'number_of_rooms')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

# Handle UserProfile with User
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    fields = ['organisation', 'name', 'email', 'phone_number']

class CustomUserAdmin(UserAdmin):
    inlines = [UserProfileInline]
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff', 'get_organisation']

    def get_organisation(self, obj):
        try:
            return obj.userprofile.organisation.name
        except:
            return "-"

    get_organisation.short_description = 'Organisation'
# Unregister default User and register custom
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)