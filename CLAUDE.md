# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Building Docker Images Locally
```bash
# Build base OpenCV image (must be built first)
./build/opencv_base.sh

# Build Python OpenCV image (depends on base image)
./build/opencv_python.sh
```

### Running Tests
```bash
# Test GPU functionality (requires NVIDIA GPU)
docker run --gpus all --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools -it ajsinclair/opencv-base python3 /test/test_gpu.py
```

## Architecture Overview

This repository creates Docker images for OpenCV with NVIDIA CUDA support. The build process is structured as a two-stage pipeline:

1. **Base Image (`opencv-base`)**: Ubuntu 24.04 with OpenCV 4.11.0 compiled from source with CUDA support, targeting NVIDIA T4 GPUs (CUDA architecture 7.5)

2. **Python Image (`opencv-python`)**: Extends the base image to build opencv-python wheels with CUDA bindings

### Key Design Decisions

- **Dynamic Python Discovery**: The base Dockerfile uses runtime detection to find the correct Python version and site-packages location, making it resilient to Python version changes
- **CUDA Architecture**: Currently hardcoded to 7.5 for T4 GPUs in the base Dockerfile (line with `-D CUDA_ARCH_BIN=7.5`)
- **Sequential Build**: Python image depends on base image being built first
- **GitHub Actions**: Automated builds trigger on pushes to main branch, publishing to Docker Hub

### Critical Files and Locations

- **Dockerfiles**: `/dockerfiles/Dockerfile_opencv_base` and `/dockerfiles/Dockerfile_opencv_python`
- **Build Scripts**: `/build/opencv_base.sh` and `/build/opencv_python.sh`
- **CI/CD**: `.github/workflows/docker-opencv.yml`
- **GPU Tests**: `/test/test_gpu.py`

### Adding New GPU Architecture Support

To support additional GPU architectures, modify the CUDA_ARCH_BIN value in `/dockerfiles/Dockerfile_opencv_base`. The TODO list includes architectures 8.0, 8.6, 8.7 (Orin/Jetson), 8.9, 9.0, 10.0, and 12.0.

### Docker Hub Publishing

Images are automatically published to:
- `ajsinclair/opencv-base:latest` and `ajsinclair/opencv-base:nvidia-cuda-7.5`
- `ajsinclair/opencv-python:latest` and `ajsinclair/opencv-python:nvidia-cuda-7.5`