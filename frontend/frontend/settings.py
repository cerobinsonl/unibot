import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'cambia-esto-por-uno-seguro'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'whitenoise.runserver_nostatic',  # <-- se recomienda poner esto primero
    'django.contrib.staticfiles',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

ROOT_URLCONF = 'frontend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'core' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {},
    },
]

WSGI_APPLICATION = 'frontend.wsgi.application'
ASGI_APPLICATION = 'frontend.asgi.application'

# DB mínima (SQLite) si no usas funcionalidades de Django que requieran DB
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

DEBUG = True
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'core' / 'static',  # Donde guardas tus CSS dentro de la app
]

