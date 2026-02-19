from django.urls import path
from . import views

urlpatterns = [
    # We'll add views later
    path('', views.home, name='home'),
]