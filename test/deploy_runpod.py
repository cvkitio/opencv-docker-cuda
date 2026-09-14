#!/usr/bin/env python3
"""
RunPod Deployment Script for OpenCV CUDA Tests

This script deploys the opencv-python-test Docker image to RunPod,
runs the GPU tests, and retrieves the results.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

try:
    import runpod
except ImportError:
    print("RunPod SDK not installed. Install with: pip install runpod")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RunPodTestRunner:
    """Handles deployment and testing on RunPod."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RunPod client with API key."""
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RunPod API key is required. Set RUNPOD_API_KEY environment variable.")
        
        runpod.api_key = self.api_key
        self.template_id = None
        self.endpoint_id = None
        
    def get_available_gpus(self) -> Dict[str, Any]:
        """Get list of available GPU types."""
        try:
            gpus = runpod.get_gpus()
            logger.info(f"Available GPUs: {len(gpus)} types")

            return gpus
        except Exception as e:
            logger.error(f"Failed to get GPU list: {e}")
            return {}
    
    def create_template(self, template_name: str = "opencv-cuda-test") -> str:
        """Create a template for the OpenCV test container."""
        template_config = {
            "name": template_name,
            "image_name": "ajsinclair/opencv-python-test:latest",
            #"container_disk_in_gb": 10,
            #"volume_in_gb": 0,
            #"ports": "8888/http",
            #"env": [
            #    {"key": "PYTHONUNBUFFERED", "value": "1"}
            #],
            #"start_ssh": False,
            #"start_jupyter": False
        }
        
        try:
            logger.info(f"Creating template: {template_name}")
            template = runpod.create_template(**template_config)
            self.template_id = template["id"]
            logger.info(f"Template created with ID: {self.template_id}")
            return self.template_id
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise
    
    def create_endpoint(self, endpoint_name: str = "opencv-test-endpoint", 
                       gpu_type: str = "NVIDIA GeForce RTX 3080 Ti") -> str:
        """Create an endpoint for running tests."""
        if not self.template_id:
            raise ValueError("Template must be created first")
        
        endpoint_config = {
            "name": endpoint_name,
            "template_id": self.template_id,
            "gpu_ids": gpu_type,
            "workers_min": 0,
            "workers_max": 1,
            "idle_timeout": 5,  # Shutdown after 5 seconds of inactivity
            "locations": "US"   # Prefer US locations
        }
        
        try:
            logger.info(f"Creating endpoint: {endpoint_name}")
            endpoint = runpod.create_endpoint(**endpoint_config)
            self.endpoint_id = endpoint["id"]
            logger.info(f"Endpoint created with ID: {self.endpoint_id}")
            return self.endpoint_id
        except Exception as e:
            logger.error(f"Failed to create endpoint: {e}")
            raise
    
    def wait_for_endpoint_ready(self, timeout: int = 300) -> bool:
        """Wait for the endpoint to be ready using health check."""
        if not self.endpoint_id:
            raise ValueError("Endpoint must be created first")
        
        logger.info("Waiting for endpoint to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                endpoint = runpod.Endpoint(self.endpoint_id)
                health_check = endpoint.health()
                
                # Check worker status
                workers = health_check.get("workers", {})
                ready_workers = workers.get("ready", 0)
                initializing_workers = workers.get("initializing", 0)
                
                logger.info(f"Workers - Ready: {ready_workers}, Initializing: {initializing_workers}")
                
                if ready_workers > 0:
                    logger.info("Endpoint is ready with available workers!")
                    return True
                elif initializing_workers == 0 and ready_workers == 0:
                    # Check if there are any workers at all
                    total_workers = sum(workers.values()) if workers else 0
                    if total_workers == 0:
                        logger.warning("No workers found, endpoint may have failed")
                        time.sleep(10)
                        continue
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.warning(f"Error checking endpoint health: {e}")
                time.sleep(10)
        
        logger.error(f"Endpoint did not become ready within {timeout} seconds")
        return False
    
    def run_test(self, test_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the OpenCV CUDA tests on the endpoint."""
        if not self.endpoint_id:
            raise ValueError("Endpoint must be created and ready")
        
        # Default test input - the container will run tests automatically
        if test_input is None:
            test_input = {
                "input": {
                    "prompt": "Run GPU tests",
                    "test_params": {
                        "run_benchmarks": True,
                        "image_sizes": ["1000,1000", "2000,2000", "4000,4000"]
                    }
                }
            }
        
        try:
            logger.info("Submitting test job to endpoint...")
            result = runpod.run_sync(
                endpoint_id=self.endpoint_id,
                request=test_input,
                timeout=600  # 10 minute timeout
            )
            
            logger.info("Test completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to run test: {e}")
            raise
    
    def cleanup(self):
        """Clean up created resources."""
        if self.endpoint_id:
            try:
                logger.info(f"Deleting endpoint: {self.endpoint_id}")
                runpod.delete_endpoint(self.endpoint_id)
                logger.info("Endpoint deleted")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint: {e}")
        
        if self.template_id:
            try:
                logger.info(f"Deleting template: {self.template_id}")
                runpod.delete_template(self.template_id)
                logger.info("Template deleted")
            except Exception as e:
                logger.warning(f"Failed to delete template: {e}")


def main():
    """Main function to run the deployment and tests."""
    # Configuration
    gpu_type = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 3080 Ti")
    template_name = os.getenv("TEMPLATE_NAME", "opencv-cuda-test")
    endpoint_name = os.getenv("ENDPOINT_NAME", "opencv-test-endpoint")
    
    runner = None
    try:
        # Initialize RunPod runner
        logger.info("Initializing RunPod test runner...")
        runner = RunPodTestRunner()
        
        # Get available GPUs
        logger.info("Checking available GPU types...")
        gpus = runner.get_available_gpus()
        if gpus:
            logger.info(f"Available GPU types: {gpus}")
        
        # Create template
        template_id = runner.create_template(template_name)
        
        # Create endpoint
        endpoint_id = runner.create_endpoint(endpoint_name, gpu_type)
        
        # Wait for endpoint to be ready
        if not runner.wait_for_endpoint_ready():
            logger.error("Endpoint failed to become ready")
            return 1
        
        # Run tests
        logger.info("Running OpenCV CUDA tests...")
        test_results = runner.run_test()
        
        # Display results
        logger.info("=== TEST RESULTS ===")
        print(json.dumps(test_results, indent=2))
        
        # Check for test success
        if test_results.get("status") == "COMPLETED":
            logger.info("Tests completed successfully!")
            return 0
        else:
            logger.error("Tests failed or did not complete properly")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1
    finally:
        if runner:
            logger.info("Cleaning up resources...")
            runner.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)