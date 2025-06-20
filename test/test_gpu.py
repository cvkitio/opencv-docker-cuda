import cv2
import dlib
import time
import torch
import numpy as np
import sys

def test_gpu_opencv():
    """
    Tests GPU availability and performance with OpenCV.
    """
    print("OpenCV GPU Test:")
    # Check if CUDA is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA is available: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) detected")
        # Print detailed information about the first GPU
        cv2.cuda.printCudaDeviceInfo(0)
    else:
        print("CUDA is not available in OpenCV. OpenCV will use CPU.")
        return  # Stop the test if no CUDA

    # Create a large random image on the CPU
    cpu_image = np.random.randint(0, 256, (4000, 4000, 3), dtype=np.uint8)

    # Upload the image to the GPU
    gpu_image = cv2.cuda_GpuMat()
    start_time = time.time()
    gpu_image.upload(cpu_image)
    upload_time = time.time() - start_time
    print(f"Time to upload image to GPU: {upload_time:.4f} seconds")

    # Perform a Gaussian blur on the GPU
    start_time = time.time()
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)
    gpu_blurred_image = gaussian_filter.apply(gpu_image)
    gpu_blur_time = time.time() - start_time
    print(f"Time to perform Gaussian blur on GPU: {gpu_blur_time:.4f} seconds")

    # Perform the same operation on the CPU
    start_time = time.time()
    cpu_blurred_image = cv2.GaussianBlur(cpu_image, (5, 5), 0)
    cpu_blur_time = time.time() - start_time
    print(f"Time to perform Gaussian blur on CPU: {cpu_blur_time:.4f} seconds")

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"Speedup: {cpu_blur_time / gpu_blur_time:.2f}x")

def test_gpu_dlib():
    """
    Tests GPU availability and performance with dlib.  This test specifically
    checks the speed of face detection.
    """
    print("\ndlib GPU Test:")
    # Check if dlib can use CUDA
    if dlib.DLIB_USE_CUDA:
        print("dlib is configured to use CUDA.")
        # Get the number of CUDA devices
        print(f"Number of CUDA devices: {dlib.cuda.get_num_devices()}")
        #Prints the name of the first device
        print(f"Device name: {dlib.cuda.get_device_name(0)}")

    else:
        print("dlib is not configured to use CUDA. dlib will use CPU.")
        return

    # Load a sample image (you can replace this with a larger image for more stress)
    try:
        image_path = "test_face.jpg" # Make sure this exists
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}.  Please make sure it exists and is a valid image file.")
            return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load dlib's face detector (this can be slow on the CPU)
    detector = dlib.get_frontal_face_detector()

    # Perform face detection on the GPU
    start_time = time.time()
    faces_gpu = detector(image, 1) # The '1' here means to upscale the image once
    gpu_time = time.time() - start_time
    print(f"Time to perform face detection on GPU: {gpu_time:.4f} seconds")

    # Perform face detection on the CPU
    start_time = time.time()
    faces_cpu = detector(image, 1)
    cpu_time = time.time() - start_time
    print(f"Time to perform face detection on CPU: {cpu_time:.4f} seconds")
    if dlib.DLIB_USE_CUDA:
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
def test_pytorch_gpu():
    """
    Tests GPU availability and performance with PyTorch.
    """
    print("\nPyTorch GPU Test:")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"PyTorch is using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch is using CPU.")
        return

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Create a large random tensor on the CPU
    cpu_tensor = torch.randn(4000, 4000)

    # Move the tensor to the GPU
    start_time = time.time()
    gpu_tensor = cpu_tensor.to('cuda')
    upload_time = time.time() - start_time
    print(f"Time to move tensor to GPU: {upload_time:.4f} seconds")

    # Perform a matrix multiplication on the GPU
    start_time = time.time()
    gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
    gpu_matmul_time = time.time() - start_time
    print(f"Time to perform matrix multiplication on GPU: {gpu_matmul_time:.4f} seconds")

    # Perform the same operation on the CPU
    start_time = time.time()
    cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
    cpu_matmul_time = time.time() - start_time
    print(f"Time to perform matrix multiplication on CPU: {cpu_matmul_time:.4f} seconds")
    if torch.cuda.is_available():
        print(f"Speedup: {cpu_matmul_time / gpu_matmul_time:.2f}x")

if __name__ == "__main__":
    # Print the system information
    print("System Information:")
    print(f"  Python Version: {sys.version}")
    print(f"  OpenCV Version: {cv2.__version__}")
    print(f"  dlib Version: {dlib.__version__}")
    print(f"  PyTorch Version: {torch.__version__}")

    # Run the tests
    test_gpu_opencv()
    test_gpu_dlib()
    test_pytorch_gpu()
