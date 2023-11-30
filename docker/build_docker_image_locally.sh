#!/usr/bin/env bash
#set -e

#######################################
# Builds and publishes torch-dag docker images
# This script takes 0 or 1 arguments:
# - <no arguments>      : behaves as if [default] was passed
# - torch-dag [default] : builds the torch-dag environment image
# - torch-dag-plus      : builds the torch-dag extended environment image
# - all                 : builds all images           
#######################################

REGISTRY="registry.gitlab.com/tcl-research/auto-ml/devine/"
TORCH_DAG_BUILD_STAGE="torch-dag"
TORCH_DAG_PLUS_BUILD_STAGE="torch-dag-plus"

TAG="0.0.9-pytorch-2.1.0-timm-0.9.5-cuda11.8-cudnn8-runtime"

TORCH_DAG_IMAGE_IDENTIFIER="${REGISTRY}${TORCH_DAG_BUILD_STAGE}:${TAG}"
TORCH_DAG_PLUS_IMAGE_IDENTIFIER="${REGISTRY}${TORCH_DAG_PLUS_BUILD_STAGE}:${TAG}"


build_and_push() {
    docker build --build-arg CACHE_DATE="$(date)" -t $1 --target $2 .
    docker push $1
}

run_torch_dag() {
    build_and_push ${TORCH_DAG_IMAGE_IDENTIFIER} ${TORCH_DAG_BUILD_STAGE}
}

run_torch_dag_plus() {
    build_and_push ${TORCH_DAG_PLUS_IMAGE_IDENTIFIER} ${TORCH_DAG_PLUS_BUILD_STAGE}
}


# Check if no arguments provided
if [ $# -eq 0 ]; then
    run_torch_dag
    exit 0
fi

first_arg="${1,,}" # Convert to lowercase for case-insensitive comparison

if [ "$first_arg" = "all" ]; then
    run_torch_dag
    run_torch_dag_plus
    exit 0
    elif [ "$first_arg" = "torch-dag" ]; then
    run_torch_dag
    exit 0
    elif [ "$first_arg" = "torch-dag-plus" ]; then
    run_torch_dag_plus
    exit 0
else
    echo "Error: No valid commands executed"
    exit 1
fi