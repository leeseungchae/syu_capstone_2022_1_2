from django.urls import include, path

urlpatterns = [path("handler/", include("Handler.urls"))]
