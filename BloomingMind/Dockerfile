FROM ubuntu:20.04
LABEL maintainer='ztmdco@gmail.com'
LABEL version='1.0.0'
LABEL description='BloomingMind Docker'
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/BloomingMind
COPY requirements.txt ./


RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN apt-get install -y wget
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN python3.9 -m pip install --upgrade pip
RUN apt-get install python3-dev libmysqlclient-dev gcc -y
RUN apt-get install -y vim

#RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . .
