FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    git \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsndfile1 \
    tar \
    unzip \
    vim \
    keychain

# Copy files from host to the image.
COPY requirements.txt /tmp/requirements.txt

# Upgrade pip and install Python packages from requirements.txt.
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Set the language settings.
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US

# Create the user.
ARG USERNAME
ARG USER_UID
ARG USER_GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
