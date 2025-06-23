#!/bin/bash

# GPU Test Runner Script
# This script runs the refactored GPU tests with pytest

echo "==================================="
echo "Running GPU Tests with pytest"
echo "==================================="

# Install test dependencies if needed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "Installing test dependencies..."
    pip install -r requirements-test.txt
fi

# Run all tests with timing
echo "Running all GPU tests..."
python3 -m pytest test_gpu_refactored.py -v

# Run specific test categories
echo -e "\n==================================="
echo "Running tests by category:"
echo "==================================="

# OpenCV tests only
echo -e "\nOpenCV Tests:"
python3 -m pytest test_gpu_refactored.py -v -m opencv

# PyTorch tests only
echo -e "\nPyTorch Tests:"
python3 -m pytest test_gpu_refactored.py -v -m pytorch

# Dlib tests only
echo -e "\nDlib Tests:"
python3 -m pytest test_gpu_refactored.py -v -m dlib

# Benchmark tests only
echo -e "\nBenchmark Tests:"
python3 -m pytest test_gpu_refactored.py -v -m benchmark

# Generate timing summary
echo -e "\n==================================="
echo "Timing Summary:"
echo "==================================="
python3 -c "
from timing_decorator import get_timing_summary
import json
summary = get_timing_summary()
if summary:
    print(json.dumps(summary, indent=2))
else:
    print('No timing data available yet.')
"

echo -e "\nTest reports saved to:"
echo "  - test_report.json (pytest report)"
echo "  - test_timings.json (timing data)"