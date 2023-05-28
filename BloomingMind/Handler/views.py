# Create your views here.
import logging
import os

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from Handler.exceptions import ImageFIleError
from ImageRecognition.service import ImageRecognizer

logger = logging.getLogger("BloomingMind")


class HandleFileUploadView(View):
    """STT View"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @csrf_exempt
    def post(self, request: HttpRequest) -> HttpResponse:
        """
        이미지 업로드 요청 API
        Args:
            request(HttpRequest): 이미지 파일이 포함된 요청

        Returns:
            JsonResponse: 성공시 (200 OK)
            JsonResponse: 잘못된 요청 (404 error)
            JsonResponse: 실패시 (402 error)

        """
        try:
            if request.FILES["image"]:
                image_file = request.FILES["image"]
                image_path = image_save(image_file=image_file)
                message = "Image upload Success"
                logger.info(message)

                image_reco = ImageRecognizer(image_path)
                message = "Get started with image analysis"
                logger.info(message)

                pred = image_reco.inference()
                message = "Finish analyzing the image"
                logger.info(message)

                if not pred:
                    pred = ""
                    message = "I'm having trouble recognizing images"

                return JsonResponse(
                    {"status_code": 200, "image_name": pred, "message": message},
                    status=200,
                )

            else:
                return JsonResponse(
                    {
                        "status_code": 404,
                        "error": "It's not an image",
                        "message": "Please upload the image",
                    },
                    status=404,
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(str(e))
            return JsonResponse(
                {"status_code": 402, "error": str(e), "message": "Upload failed"},
                status=402,
            )


def image_conversion(image: str):
    """
    이미지 파일을 변환 하는 함수
    Args:
        image(str): 이미지 파일

    Returns:
        image_bytes
    """
    from io import BytesIO

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return image_bytes


def is_image_file(image_name):
    """
    파일 확장자 검사 하는 함수
    Args:
        image_name(str): 이미지 파일 이름

    Returns:
        True or False
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"]

    # print(os.path.splitext(str(image_name.filename)))
    _, extension = os.path.splitext(image_name)
    if str(extension).lower() in image_extensions:
        return True
    else:
        return False

    # return extension.lower() in image_extensions


def image_save(image_file: str) -> str:
    """
    업로드 된 이미지를 저장하는 함수
    Args:
        image_file(str): byte 으로 이루 어진 이미지 파일

    Returns:
        image_path(str): 이미지 파일이 저장된 path (uuid)
    """
    import uuid

    from BloomingMind.settings.base import ROOT_DIR

    image_name = str(image_file)
    filecheck = is_image_file(image_name=image_name)
    if not filecheck:
        raise ImageFIleError("This file cannot be read.")

    uuid_name = uuid.uuid4()
    save_path = os.path.join(ROOT_DIR, "data", "image")
    os.makedirs(save_path, exist_ok=True)
    try:
        with open(
            os.path.join(save_path, str(uuid_name) + ".jpg"), "wb+"
        ) as image_write:
            if image_name.endswith(("jp", "jpeg")):
                image_bytes = image_conversion(image_file)
                image_write.write(image_bytes)
            else:
                for chunk in image_file.chunks():
                    image_write.write(chunk)
            return os.path.join(save_path, str(uuid_name) + ".jpg")
    except:
        raise ImageFIleError("This file cannot be read.")
