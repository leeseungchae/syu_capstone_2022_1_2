from django.urls import path

from .views import HandleFileUploadView

urlpatterns = [
    path("image-upload", HandleFileUploadView.as_view()),
]
