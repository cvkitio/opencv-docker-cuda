docker buildx build --platform linux/amd64 -f dockerfiles/Dockerfile_opencv_base -t ajsinclair/opencv-base:latest --load .
docker tag ajsinclair/opencv-base ajsinclair/opencv-base:arch7.5
docker push ajsinclair/opencv-base:latest