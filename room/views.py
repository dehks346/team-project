from django.shortcuts import render


# Create your views here.
def home(request):
    # Render the dashboard template; static files are handled by Django
    return render(request, "index.html")