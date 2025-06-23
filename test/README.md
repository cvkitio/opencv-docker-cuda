# GPU Test Suite

This directory contains refactored GPU tests for OpenCV, dlib, and PyTorch with CUDA support.

## Test Structure

The test suite has been refactored from a single monolithic script into a modular pytest-based framework:

- `test_gpu_refactored.py` - Main test file with individual test functions
- `timing_decorator.py` - Custom decorator for logging test execution times
- `requirements-test.txt` - Test dependencies
- `pytest.ini` - pytest configuration
- `run_tests.sh` - Test runner script

## Running Tests

### Quick Start
```bash
./run_tests.sh
```

### Manual pytest Commands
```bash
# Run all tests
pytest test_gpu_refactored.py -v

# Run only OpenCV tests
pytest test_gpu_refactored.py -v -m opencv

# Run only PyTorch tests
pytest test_gpu_refactored.py -v -m pytorch

# Run only dlib tests
pytest test_gpu_refactored.py -v -m dlib

# Run only benchmark tests
pytest test_gpu_refactored.py -v -m benchmark

# Run tests for a specific image/matrix size
pytest test_gpu_refactored.py -v -k "1000, 1000"
```

### Docker Usage
```bash
docker run --gpus all \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  -v $(pwd)/test:/test \
  -it ajsinclair/opencv-base \
  bash -c "cd /test && ./run_tests.sh"
```

## Test Features

### Timing Decorator
All tests use the `@log_timing()` decorator which:
- Measures test execution time
- Logs results to `test_timings.json`
- Captures metadata like GPU/CPU times and speedup ratios

### Parametrized Tests
Performance tests run with multiple sizes:
- Image sizes: 1000x1000, 2000x2000, 4000x4000
- Matrix sizes: 1000x1000, 2000x2000, 4000x4000

### Test Markers
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.opencv` - OpenCV-specific tests
- `@pytest.mark.pytorch` - PyTorch-specific tests
- `@pytest.mark.dlib` - dlib-specific tests
- `@pytest.mark.benchmark` - Performance comparison tests

### Skip Conditions
Tests automatically skip when:
- No CUDA devices are available
- Required libraries don't have CUDA support compiled in

## Output Files

- `test_report.json` - Detailed pytest results in JSON format
- `test_timings.json` - Historical timing data for all test runs

## Legacy Tests

The original monolithic test script is preserved as `test_gpu.py` for reference.