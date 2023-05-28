"""
ASGI config for BloomingMind project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BloomingMind.settings.dev")

application = get_asgi_application()


# from channels.auth import AuthMiddlewareStack
# from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.security.websocket import AllowedHostsOriginValidator
# from django.core.asgi import get_asgi_application
#
# import Handler.routing

# application = ProtocolTypeRouter({
#   "http": get_asgi_application(),
#   "websocket": AllowedHostsOriginValidator(
#         AuthMiddlewareStack(
#             URLRouter(
#                 Handler.routing.websocket_urlpatterns	# chat 은 routing.py 가 들어있는 앱 이름
#             )
#         )
#     ),
# })
