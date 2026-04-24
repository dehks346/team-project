from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django import forms
from .models import Room
from django import forms
from django.utils import timezone
from .models import Booking, Room

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

class RoomForm(forms.ModelForm):
    class Meta:
        model = Room
        fields = ['location', 'capacity', 'room_type', 'is_active']
        widgets = {
            'location': forms.TextInput(attrs={
                'class': '',
                'placeholder': 'e.g. Conference Room B',
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 1rem;'
            }),
            'capacity': forms.NumberInput(attrs={
                'class': '',
                'min': 1,
                'max': 50,
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text);'
            }),
            'room_type': forms.Select(attrs={
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text);'
            }),
            'is_active': forms.CheckboxInput(attrs={'style': 'accent-color: var(--primary); width: 20px; height: 20px;'}),
        }

class BookingForm(forms.ModelForm):
    class Meta:
        model = Booking
        fields = ['room', 'booking_datetime']
        widgets = {
            'room': forms.Select(attrs={
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 1rem;',
            }),
            'booking_datetime': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text);',
            }),
        }

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user and hasattr(user, 'userprofile'):
            # Only show rooms from the user's organisation
            self.fields['room'].queryset = Room.objects.filter(
                organisation=user.userprofile.organisation,
                is_active=True
            ).order_by('location')

class BookingEditForm(forms.ModelForm):
    class Meta:
        model = Booking
        fields = ['booking_datetime', 'status']
        widgets = {
            'booking_datetime': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text);'
            }),
            'status': forms.Select(attrs={
                'style': 'width: 100%; padding: 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text);'
            }),
        }