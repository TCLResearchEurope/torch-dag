#!/usr/bin/env bash
#set -e
IMAGE_IDENTIFIER="registry.gitlab.com/tcl-research/auto-ml/devine/torch-dag:0.0.9-pytorch-2.1.0-timm-0.9.5-cuda11.8-cudnn8-runtime"
docker build --build-arg CACHE_DATE="$(date)" -t ${IMAGE_IDENTIFIER} .
#docker build -t ${IMAGE_IDENTIFIER} .
docker push ${IMAGE_IDENTIFIER}


