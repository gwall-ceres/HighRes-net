# Use a recent CUDA base image with cuDNN 8 and Ubuntu 20.04
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies, add Deadsnakes PPA, and install Python 3.7 with distutils
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3-pip \
    git \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Upgrade pip for Python 3.7
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements.txt into the image
COPY requirements.txt /app/requirements.txt

# Install numpy first to satisfy scikit-learn dependency
RUN python3 -m pip install --no-cache-dir numpy

# Install the remaining Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /app

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables if needed
ENV PYTHONUNBUFFERED=1

# Specify the default command (modify as needed)
CMD ["python3", "train.py"]
