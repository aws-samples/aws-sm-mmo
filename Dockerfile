# ==================================================================
# Purpose: Serve Multiple Models with One Amazon Sagamaker Endpoint
# ------------------------------------------------------------------
# Author : Zheng Zhang
# Date : March, 2022
# ==================================================================

FROM nvcr.io/nvidia/tritonserver:22.04-py3

LABEL maintainer="Zheng Zhang"


# Add arguments to achieve the version, python and url
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

ENV LANG=C.UTF-8

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
 && apt-get update \
 && apt-get install -y nginx \
 && apt-get install -y libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN ${PIP} install -U --no-cache-dir \
tritonclient[all] \
torch \
torchvision \
pillow==9.1.1 \
scipy==1.8.1 \
transformers==4.20.1 \
opencv-python==4.6.0.66 \
flask \
gunicorn \
&& \

ldconfig && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/* ~/* &&\
mkdir -p /opt/program/models/ 


ENV PATH="/opt/program:/bin:${PATH}"

# Set up the program in the image
COPY sm /opt/program
COPY model /opt/program/models
WORKDIR /opt/program

ENTRYPOINT ["python3", "serve"]
