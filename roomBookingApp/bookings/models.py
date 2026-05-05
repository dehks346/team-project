from django.db import models

# Create your models here.
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver

class Organisation(models.Model):
    """Entity: Organisation"""

    # Primary Key
    organisation_id = models.AutoField(primary_key=True)

    # Basic Info
    name = models.CharField(max_length=200, unique=True)
    email_address = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Store password
    unique_access_code = models.CharField(max_length=50, unique=True, help_text="Unique code for organisation access")
    phone_number = models.CharField(max_length=20)
    address = models.TextField()

    # Billing
    fee = models.DecimalField(max_digits=10, decimal_places=2, help_text="Monthly fee")

    # Stats (can be calculated, but stored for quick access)
    number_of_users = models.PositiveIntegerField(default=0, help_text="Total users in organisation")
    number_of_rooms = models.PositiveIntegerField(default=0, help_text="Total rooms owned by organisation")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']


class Room(models.Model):
    """Entity Room"""
    ROOM_TYPES = [('CONF', 'Conference Room'), ('MEET', 'Meeting Room'), ('TRAIN', 'Training Room')]

    """Primary Key"""
    room_id = models.AutoField(primary_key=True)
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE, related_name='rooms', null=True,
                                     blank=True)

    name = models.CharField(max_length=200, default='Unnamed Room')

    location = models.CharField(max_length=200)
    capacity = models.PositiveIntegerField()
    room_type = models.CharField(max_length=50, choices=ROOM_TYPES, default='MEET')

    is_face_required = models.BooleanField(default=False)
    approval_required = models.BooleanField(default=False)
    equipment = models.JSONField(default=list, blank=True)
    opening_time = models.TimeField(null=True, blank=True)
    closing_time = models.TimeField(null=True, blank=True)
    photo_name = models.CharField(max_length=255, blank=True)

    """Status"""
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def active_bookings_count(self):
        """Number of confirmed future bookings"""
        return self.bookings.filter(
            booking_datetime__gte=timezone.now(),
            status='CONFIRMED'
        ).count()

    @property
    def occupancy_display(self):
        """to show usage of slots used for bookings"""
        return f"{self.active_bookings_count}/{self.capacity}"

    @property
    def is_available(self):
        """to check if the room can accept more bookings"""
        return self.active_bookings_count < self.capacity

    @property
    def availability_status(self):
        """returns colored status for UI Purposes"""
        if not self.is_active:
            return "Inactive"
        elif self.active_bookings_count >= self.capacity:
            return "Fully Booked"
        elif self.active_bookings_count >= self.capacity * 0.7:
            return "Nearly Full"
        else:
            return "Available"

    def __str__(self):
        return f"{self.name} - {self.location} ({self.room_type})"

    class Meta:
        ordering = ['room_id']


class UserProfile(models.Model):
    """Entity UserProfile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE, related_name='users', null=True,
                                     blank=True)

    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=20)

    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def total_bookings(self):
        """bookings made by user (total)"""
        return self.bookings.count()

    @property
    def upcoming_bookings(self):
        return self.bookings.filter(booking_datetime__gte=timezone.now(), status='CONFIRMED').count()

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']

# MOVED Record class ABOVE Booking to fix import issue
class Record(models.Model):
    """Entity 1: Records - Audit log for all booking activities"""
    ACTION_TYPES = [
        ('BOOKING_CREATED', 'Booking Created'),
        ('BOOKING_CANCELLED', 'Booking Cancelled'),
        ('BOOKING_COMPLETED', 'Booking Completed'),
        ('ACCESS_ATTEMPT', 'Access Attempt'),
        ('USER_ADDED', 'User Added'),
        ('ROOM_MODIFIED', 'Room Modified'),
    ]

    # Primary Key
    record_number = models.AutoField(primary_key=True)

    # Related objects (optional - some records might not have all)
    booking = models.ForeignKey('Booking', on_delete=models.SET_NULL, null=True, blank=True, related_name='records')
    room = models.ForeignKey(Room, on_delete=models.SET_NULL, null=True, blank=True, related_name='records')
    user = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name='records')

    # Record Details
    action = models.CharField(max_length=50, choices=ACTION_TYPES)
    description = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Store additional data as JSON
    details = models.JSONField(default=dict, blank=True)

    @property
    def summary(self):
        """Short summary of the record"""
        if self.booking:
            return f"{self.action}: {self.booking}"
        elif self.room:
            return f"{self.action}: {self.room}"
        elif self.user:
            return f"{self.action}: {self.user}"
        else:
            return f"{self.action} at {self.timestamp}"

    def __str__(self):
        return f"Record #{self.record_number}: {self.action}"

    class Meta:
        ordering = ['-timestamp']


class Booking(models.Model):
    """Entity Booking"""
    BOOKING_STATUS = [
        ('CONFIRMED', 'Confirmed'),
        ('CANCELLED', 'Cancelled'),
        ('COMPLETED', 'Completed'),
        ('NO_SHOW', 'No Show'),
    ]

    booking_id = models.AutoField(primary_key=True)

    """Foreign Keys"""
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='bookings')
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='bookings')

    booking_datetime = models.DateTimeField()
    status = models.CharField(max_length=50, choices=BOOKING_STATUS, default='CONFIRMED')
    duration_minutes = models.PositiveIntegerField(default=60)
    notes = models.TextField(blank=True)
    is_recurring = models.BooleanField(default=False)
    recurrence_pattern = models.CharField(max_length=50, blank=True)

    """Timestamps"""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Validation
    def clean(self):
        if self.booking_datetime and self.booking_datetime < timezone.now():
            raise ValidationError('Booking time must be in the future')

        # Check room capacity for new bookings
        if not self.pk and self.status == 'CONFIRMED':
            if not self.room.is_available:
                raise ValidationError(f'Room is at capacity ({self.room.occupancy_display})')

    def save(self, *args, **kwargs):
        is_new = self._state.adding
        self.clean()
        super().save(*args, **kwargs)

        # Create a record entry automatically when booking is created
        if is_new:
            Record.objects.create(
                booking=self,
                action='BOOKING_CREATED',
                description=f"Booking created for {self.room.location}"
            )

    @property
    def is_upcoming(self):
        """Check if booking is in the future"""
        return self.booking_datetime > timezone.now()

    @property
    def can_cancel(self):
        """Check if booking can be canceled"""
        return (self.status == 'CONFIRMED' and
                self.booking_datetime > timezone.now())

    def __str__(self):
        return f"Booking #{self.booking_id}: {self.user.name} @ {self.room.location}"

    class Meta:
        ordering = ['-booking_datetime']


class BookingInvitation(models.Model):
    """Entity for tracking invite decisions per invited user."""

    INVITE_STATUS = [
        ('PENDING', 'Pending'),
        ('ACCEPTED', 'Accepted'),
        ('DECLINED', 'Declined'),
    ]

    invitation_id = models.AutoField(primary_key=True)
    booking = models.ForeignKey(Booking, on_delete=models.CASCADE, related_name='invitations')
    invited_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='booking_invitations')
    invited_email = models.EmailField(blank=True)
    invited_name = models.CharField(max_length=150, blank=True)
    status = models.CharField(max_length=20, choices=INVITE_STATUS, default='PENDING')
    responded_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def mark_accepted(self):
        self.status = 'ACCEPTED'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

    def mark_declined(self):
        self.status = 'DECLINED'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

    @property
    def display_name(self):
        if self.invited_user:
            return self.invited_user.get_full_name().strip() or self.invited_user.username
        return self.invited_name or self.invited_email

    def __str__(self):
        return f"Invite #{self.invitation_id}: {self.display_name} - {self.status}"

    class Meta:
        ordering = ['-created_at']


class Access(models.Model):
    """Entity 5: Access - Track door/room access attempts"""
    ACCESS_RESULT = [
        ('GRANTED', 'Granted'),
        ('DENIED', 'Denied'),
    ]

    # Primary Key
    access_id = models.AutoField(primary_key=True)

    # Foreign Keys
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='access_attempts')
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='access_attempts')

    # Access Details
    result = models.CharField(max_length=20, choices=ACCESS_RESULT)
    access_datetime = models.DateTimeField(auto_now_add=True)

    @property
    def time_ago(self):
        """Human readable time difference"""
        delta = timezone.now() - self.access_datetime
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600} hours ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60} minutes ago"
        else:
            return f"{delta.seconds} seconds ago"

    def __str__(self):
        return f"Access #{self.access_id}: {self.result} - {self.user.name}"

    class Meta:
        ordering = ['-access_datetime']
        verbose_name_plural = "Access"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    # Skip profile creation while loading fixtures (`loaddata`) to avoid conflicts.
    if kwargs.get('raw', False):
        return

    if created and not hasattr(instance, 'userprofile'):
        UserProfile.objects.create(
            user=instance,
            name=instance.get_full_name().strip() or instance.username,
            email=instance.email or f'{instance.username}@example.com',
            phone_number='',
        )


