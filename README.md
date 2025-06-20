# Docker images and builders for opencv cuda

A series of docker builders for various flavors of opencv versions with Nvidia hardware and CUDA versions
for opencv and opencv-python

## Versions supported
opencv
* 4.11.0

nvidia hardware
See https://developer.nvidia.com/cuda-gpus for further details

* T4 (Ampere) ARCH 7.5
* TODO 
    * 8.0
    * 8.6
    * 8.7 (Orin / Jetson)
    * 8.9
    * 9.0
    * 10.0
    * 12.0 

ffmpeg
* base off the work here and include with opencv, may be possible to use a docker layer

* TODO Additional linux distros

TODO - update this automatically from build scripts

## Prequisites

Build - docker installed

Test - Nvidia GPU instance

## Images are published to DockerHub

https://hub.docker.com/r/ajsinclair/opencv-base

To use:

`docker pull ajsinclair/opencv-base`

## Build notes

* Base opencv CUDA build options
* Extended features such as cuDNN
* Building the python modules and linking
* Compiling ffmpeg with cuda support for use (do this in a separate repo)
