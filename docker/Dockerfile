FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS torch-dag
MAINTAINER Łukasz Treszczotko <lukasz.treszczotko@tcl.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install -y git \
    software-properties-common \
    build-essential \
    curl \
    libsm6 libxext6 libxrender-dev \
    zsh \
    sudo \
    vim \
    htop \
    tmux \
    wget \
    adb \
    apt-utils \
    zip \
    gettext-base \
    graphviz \
    ffmpeg \
    jq && \
    rm -rfv /var/lib/apt/lists/*


WORKDIR /opt
COPY requirements-docker.txt requirements.txt

ARG CACHE_DATE
RUN pip --no-cache-dir install -r requirements.txt
RUN # Create user user "tclbot" with password "tcl". User will be have sudo permissions so will be able to install new stuff on the fly.
RUN useradd -ms /bin/bash tclbot; usermod -aG users,sudo,adm tclbot; echo "tclbot:tcl" | chpasswd
USER root
RUN mkdir -p /opt
RUN chown tclbot /opt
RUN echo 'export PS1="\[\e[0;31m\]\u\[\e[m\] \[\e[0;32m\]\h\[\e[m\]@\[\e[0;34m\]\w\[\e[m\]\$ "' >> /home/tclbot/.bashrc

# Now, switch to the new user
USER tclbot

FROM torch-dag as torch-dag-plus
# install DALI
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110