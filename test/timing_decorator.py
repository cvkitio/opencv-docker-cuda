import functools
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict


class TimingLogger:
    """Handles logging of test execution times to a JSON file."""
    
    def __init__(self, log_file: str = "test_timings.json"):
        self.log_file = Path(log_file)
        self._ensure_log_file()
    
    def _ensure_log_file(self) -> None:
        """Ensure the log file exists and contains valid JSON."""
        if not self.log_file.exists():
            self.log_file.write_text("[]")
        else:
            try:
                with open(self.log_file, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, IOError):
                self.log_file.write_text("[]")
    
    def log_timing(self, test_name: str, duration: float, metadata: Dict[str, Any] = None) -> None:
        """Log a test timing entry to the JSON file."""
        entry = {
            "test_name": test_name,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Read existing entries
        with open(self.log_file, 'r') as f:
            entries = json.load(f)
        
        # Append new entry
        entries.append(entry)
        
        # Write back to file
        with open(self.log_file, 'w') as f:
            json.dump(entries, f, indent=2)


# Global logger instance
_timing_logger = TimingLogger()


def log_timing(metadata_keys: list = None):
    """
    Decorator that logs the execution time of a test function.
    
    Args:
        metadata_keys: List of attribute names to extract from the test result
                      and include in the timing log metadata.
    
    Example:
        @log_timing(metadata_keys=['speedup', 'gpu_time', 'cpu_time'])
        def test_gpu_performance():
            # ... test code ...
            return {'speedup': 2.5, 'gpu_time': 0.1, 'cpu_time': 0.25}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Execute the test function
            result = func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract metadata if requested
            metadata = {}
            if metadata_keys and isinstance(result, dict):
                for key in metadata_keys:
                    if key in result:
                        metadata[key] = result[key]
            
            # Add general metadata
            metadata.update({
                "function": func.__name__,
                "module": func.__module__,
            })
            
            # Log the timing
            _timing_logger.log_timing(
                test_name=func.__name__,
                duration=duration,
                metadata=metadata
            )
            
            return result
        
        return wrapper
    return decorator


def get_timing_summary(log_file: str = "test_timings.json") -> Dict[str, Any]:
    """
    Read and summarize timing data from the log file.
    
    Returns:
        Dictionary containing summary statistics for each test.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return {}
    
    with open(log_path, 'r') as f:
        entries = json.load(f)
    
    summary = {}
    for entry in entries:
        test_name = entry['test_name']
        if test_name not in summary:
            summary[test_name] = {
                'count': 0,
                'total_duration': 0,
                'min_duration': float('inf'),
                'max_duration': 0,
                'durations': []
            }
        
        duration = entry['duration_seconds']
        summary[test_name]['count'] += 1
        summary[test_name]['total_duration'] += duration
        summary[test_name]['min_duration'] = min(summary[test_name]['min_duration'], duration)
        summary[test_name]['max_duration'] = max(summary[test_name]['max_duration'], duration)
        summary[test_name]['durations'].append(duration)
    
    # Calculate averages
    for test_name, stats in summary.items():
        stats['average_duration'] = stats['total_duration'] / stats['count']
        # Remove the raw durations list for cleaner output
        del stats['durations']
    
    return summary