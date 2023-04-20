from django.shortcuts import render

# Create your views here.
import json
import logging
from typing import Union
import os
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from .dto import ImageUploadDto

logger = logging.getLogger(__name__)


class HandleFileUploadView(View):
    """ STT View """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def get(self, _: HttpRequest) -> JsonResponse:
        """
        현재 서버에서 돌고있는 Request 정보를 리턴하는 API
        Args:
            _(HttpRequest): request

        Returns:
            JsonResponse: 성공시 (200 OK)

        """

    @csrf_exempt
    def post(self, request: HttpRequest) -> HttpResponse:
        """
        이미지 업로드 요청 API
        Args:
            request(HttpRequest): 이미지 정보를

        Returns:
            HttpResponse: 성공시 (200 OK)

        """
        try:
            if request.FILES['image']:
                upload_image = request.FILES['image']
                age = request.POST.get('age')
                gender = request.POST.get('gender')
                upload_info = ImageUploadDto(age, gender)
                image_save(upload_image)
                message = 'Image upload Success'
                logger.info(message)
                return HttpResponse(status=200)

            else:
                return JsonResponse(
                    {
                        "status_code": 404,
                        "error": "It's not an image",
                        "message": "Please upload the image"
                    }, status=404)
        except Exception as e:
            return JsonResponse(
                {
                    "status_code": 404,
                    "error": {e},
                    "message": "Upload failed"
                }, status=404)


def image_conversion(image):
    from io import BytesIO
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    return image_bytes


def image_save(upload_image):
    import uuid
    from PIL import Image
    from BloomingMind.settings.base import ROOT_DIR

    try:
        image = Image.open(upload_image)
    except:
        from .exceptions import ImageFIleError
        raise ImageFIleError('This file cannot be read.')

    image_name = uuid.uuid4()
    save_path = os.path.join(ROOT_DIR, 'data', 'image')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, str(image_name) + '.jpg'), 'wb+') as image_write:

        if 'jp' not in upload_image.name:
            image_bytes = image_conversion(image)
            image_write.write(image_bytes)
        else:
            for chunk in upload_image.chunks():
                image_write.write(chunk)
