# BloomingMind

BloomingMind version1 입니다.<br>
학습 및 API (djnago) 코드가 포함되어 있습니다.

    학습 코드 실행방법 
    
    1. cd mobilev2
    2. pip install -r requirements.txt
    1. [데이터 준비] data/original 폴더를 만들고, 하위 폴더에 꽃 이름별로 폴더를 만드세요.
    2. [데이터 전처리] image_preprocessing.py 실행을 합니다.
    3. [모델 학습] train.py 실행을 합니다
    4. [모델 테스트] model_test.py 실행을 합니다.
    5. [추론] inference.py 실행을 합니다.
-- -  

    API 서버 실행 방법 (기본)
    
    1. cd BloomingMind
    2. pip install -r requirements.txt
    3. gunicorn BloomingMind.asgi.dev:application -b 0.0.0.0:0000 -w 1 -k uvicorn.workers.UvicornWorker --reload
-- - 
    API 서버 실행 방법 (docker)
    1. docker-compose -f docker-compose-dev.yaml up
-- -
