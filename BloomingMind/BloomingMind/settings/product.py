from typing import List

from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ["frontend"]

INSTALLED_APPS += []

MIDDLEWARE += []

WSGI_APPLICATION = "BloomingMind.asgi.product.application"

# DATABASES = env("MAIN_DB")
