import cv2
import dlib
import time
import torch
import numpy as np
import sys
import pytest
from pathlib import Path
from typing import Dict, Tuple

from timing_decorator import log_timing


# Fixtures
@pytest.fixture
def gpu_available_opencv():
    """Check if OpenCV CUDA is available."""
    return cv2.cuda.getCudaEnabledDeviceCount() > 0


@pytest.fixture
def gpu_available_dlib():
    """Check if dlib CUDA is available."""
    return dlib.DLIB_USE_CUDA


@pytest.fixture
def gpu_available_pytorch():
    """Check if PyTorch CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def test_image(size: Tuple[int, int] = (4000, 4000)) -> np.ndarray:
    """Generate a test image of specified size."""
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)


@pytest.fixture
def test_tensor(size: Tuple[int, int] = (4000, 4000)) -> torch.Tensor:
    """Generate a test tensor of specified size."""
    return torch.randn(size)


# OpenCV Tests
class TestOpenCVGPU:
    
    @pytest.mark.opencv
    @pytest.mark.gpu
    @log_timing()
    def test_opencv_cuda_availability(self):
        """Test if CUDA is available in OpenCV."""
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        assert cuda_count >= 0, "CUDA device count should be non-negative"
        
        if cuda_count > 0:
            # Test that we can get device info without errors
            cv2.cuda.printCudaDeviceInfo(0)
        
        return {"cuda_devices": cuda_count}
    
    @pytest.mark.opencv
    @pytest.mark.gpu
    @pytest.mark.skipif(not cv2.cuda.getCudaEnabledDeviceCount(), reason="No CUDA devices available")
    @log_timing(metadata_keys=['upload_time', 'image_size'])
    def test_opencv_gpu_upload(self, test_image):
        """Test uploading image to GPU memory."""
        gpu_image = cv2.cuda_GpuMat()
        
        start_time = time.time()
        gpu_image.upload(test_image)
        upload_time = time.time() - start_time
        
        # Verify upload succeeded
        assert gpu_image.size() == test_image.shape[:2][::-1], "GPU image size should match CPU image"
        
        return {
            "upload_time": upload_time,
            "image_size": test_image.shape
        }
    
    @pytest.mark.opencv
    @pytest.mark.gpu
    @pytest.mark.benchmark
    @pytest.mark.skipif(not cv2.cuda.getCudaEnabledDeviceCount(), reason="No CUDA devices available")
    @pytest.mark.parametrize("image_size", [(1000, 1000), (2000, 2000), (4000, 4000)])
    @log_timing(metadata_keys=['gpu_time', 'image_size'])
    def test_opencv_gaussian_blur_gpu(self, image_size):
        """Test Gaussian blur performance on GPU."""
        # Create test image
        cpu_image = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        
        # Upload to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(cpu_image)
        
        # Create Gaussian filter
        gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)
        
        # Perform blur on GPU
        start_time = time.time()
        gpu_blurred = gaussian_filter.apply(gpu_image)
        gpu_time = time.time() - start_time
        
        # Verify result
        assert gpu_blurred.size() == gpu_image.size(), "Output size should match input"
        
        return {
            "gpu_time": gpu_time,
            "image_size": image_size
        }
    
    @pytest.mark.opencv
    @pytest.mark.gpu
    @pytest.mark.benchmark
    @pytest.mark.parametrize("image_size", [(1000, 1000), (2000, 2000), (4000, 4000)])
    @log_timing(metadata_keys=['gpu_time', 'cpu_time', 'speedup', 'image_size'])
    def test_opencv_blur_comparison(self, image_size):
        """Compare CPU vs GPU Gaussian blur performance."""
        # Create test image
        cpu_image = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        
        # CPU blur
        start_time = time.time()
        cpu_blurred = cv2.GaussianBlur(cpu_image, (5, 5), 0)
        cpu_time = time.time() - start_time
        
        # GPU blur (if available)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(cpu_image)
            
            gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)
            
            start_time = time.time()
            gpu_blurred = gaussian_filter.apply(gpu_image)
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
        else:
            gpu_time = None
            speedup = None
        
        return {
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup,
            "image_size": image_size
        }


# Dlib Tests
class TestDlibGPU:
    
    @pytest.mark.dlib
    @pytest.mark.gpu
    @log_timing()
    def test_dlib_cuda_availability(self):
        """Test if CUDA is available in dlib."""
        cuda_available = dlib.DLIB_USE_CUDA
        
        if cuda_available:
            num_devices = dlib.cuda.get_num_devices()
            assert num_devices > 0, "Should have at least one CUDA device"
            
            # Test getting device name
            device_name = dlib.cuda.get_device_name(0)
            assert device_name, "Device name should not be empty"
            
            return {
                "cuda_available": cuda_available,
                "num_devices": num_devices,
                "device_name": device_name
            }
        
        return {"cuda_available": cuda_available}
    
    @pytest.mark.dlib
    @pytest.mark.gpu
    @pytest.mark.skipif(not dlib.DLIB_USE_CUDA, reason="dlib CUDA not available")
    @log_timing(metadata_keys=['detection_time', 'num_faces'])
    def test_dlib_face_detection_gpu(self):
        """Test face detection on GPU."""
        # Create a dummy image for testing
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Load face detector
        detector = dlib.get_frontal_face_detector()
        
        # Perform detection
        start_time = time.time()
        faces = detector(test_image, 1)
        detection_time = time.time() - start_time
        
        return {
            "detection_time": detection_time,
            "num_faces": len(faces)
        }
    
    @pytest.mark.dlib
    @pytest.mark.gpu
    @pytest.mark.benchmark
    @log_timing(metadata_keys=['gpu_time', 'cpu_time', 'speedup'])
    def test_dlib_face_detection_comparison(self):
        """Compare CPU vs GPU face detection performance."""
        # Try to load a real test image
        image_path = Path("test_face.jpg")
        if image_path.exists():
            image = cv2.imread(str(image_path))
        else:
            # Create dummy image if test image doesn't exist
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            pytest.skip(f"Test image {image_path} not found, using dummy image")
        
        detector = dlib.get_frontal_face_detector()
        
        # Note: dlib's face detector doesn't have separate CPU/GPU modes
        # It automatically uses GPU if available when compiled with CUDA
        start_time = time.time()
        faces = detector(image, 1)
        detection_time = time.time() - start_time
        
        return {
            "gpu_time": detection_time if dlib.DLIB_USE_CUDA else None,
            "cpu_time": detection_time if not dlib.DLIB_USE_CUDA else None,
            "speedup": None  # Can't compare without both CPU and GPU times
        }


# PyTorch Tests
class TestPyTorchGPU:
    
    @pytest.mark.pytorch
    @pytest.mark.gpu
    @log_timing()
    def test_pytorch_cuda_availability(self):
        """Test if CUDA is available in PyTorch."""
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            assert device_count > 0, "Should have at least one CUDA device"
            
            device_name = torch.cuda.get_device_name(0)
            assert device_name, "Device name should not be empty"
            
            return {
                "cuda_available": cuda_available,
                "device_count": device_count,
                "device_name": device_name
            }
        
        return {"cuda_available": cuda_available}
    
    @pytest.mark.pytorch
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA not available")
    @pytest.mark.parametrize("tensor_size", [(1000, 1000), (2000, 2000), (4000, 4000)])
    @log_timing(metadata_keys=['upload_time', 'tensor_size'])
    def test_pytorch_tensor_upload(self, tensor_size):
        """Test uploading tensor to GPU memory."""
        cpu_tensor = torch.randn(tensor_size)
        
        start_time = time.time()
        gpu_tensor = cpu_tensor.to('cuda')
        torch.cuda.synchronize()  # Ensure upload is complete
        upload_time = time.time() - start_time
        
        # Verify upload succeeded
        assert gpu_tensor.device.type == 'cuda', "Tensor should be on GPU"
        assert gpu_tensor.shape == cpu_tensor.shape, "Shape should be preserved"
        
        return {
            "upload_time": upload_time,
            "tensor_size": tensor_size
        }
    
    @pytest.mark.pytorch
    @pytest.mark.gpu
    @pytest.mark.benchmark
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch CUDA not available")
    @pytest.mark.parametrize("matrix_size", [(1000, 1000), (2000, 2000), (4000, 4000)])
    @log_timing(metadata_keys=['gpu_time', 'matrix_size'])
    def test_pytorch_matmul_gpu(self, matrix_size):
        """Test matrix multiplication performance on GPU."""
        # Create test tensor on GPU
        gpu_tensor = torch.randn(matrix_size, device='cuda')
        
        # Warm up
        _ = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        
        # Measure performance
        start_time = time.time()
        result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # Verify result
        assert result.shape == matrix_size, "Result shape should match input"
        
        return {
            "gpu_time": gpu_time,
            "matrix_size": matrix_size
        }
    
    @pytest.mark.pytorch
    @pytest.mark.gpu
    @pytest.mark.benchmark
    @pytest.mark.parametrize("matrix_size", [(1000, 1000), (2000, 2000), (4000, 4000)])
    @log_timing(metadata_keys=['gpu_time', 'cpu_time', 'speedup', 'matrix_size'])
    def test_pytorch_matmul_comparison(self, matrix_size):
        """Compare CPU vs GPU matrix multiplication performance."""
        cpu_tensor = torch.randn(matrix_size)
        
        # CPU matmul
        start_time = time.time()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        
        # GPU matmul (if available)
        if torch.cuda.is_available():
            gpu_tensor = cpu_tensor.to('cuda')
            
            # Warm up
            _ = torch.matmul(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()
            
            start_time = time.time()
            gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
        else:
            gpu_time = None
            speedup = None
        
        return {
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup,
            "matrix_size": matrix_size
        }


# System Information Test
@pytest.mark.order(-1)  # Run first
def test_system_information():
    """Display system information."""
    print("\n" + "="*50)
    print("System Information:")
    print(f"  Python Version: {sys.version}")
    print(f"  OpenCV Version: {cv2.__version__}")
    print(f"  dlib Version: {dlib.__version__}")
    print(f"  PyTorch Version: {torch.__version__}")
    print("="*50 + "\n")