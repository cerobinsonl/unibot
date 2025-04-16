from django.urls import path, include

urlpatterns = [
    path('', include('core.urls')),  # Enlazamos las rutas de core
]
