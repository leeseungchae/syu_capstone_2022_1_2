from django.urls import path, include

urlpatterns = [
    path('handler/', include('Handler.urls'))
]
