"""
Memory Monitoring Utility for Jetson Nano/Orin Nano
Tracks CPU and GPU memory usage for optimization
"""

import psutil
import os
import gc
from pathlib import Path


class MemoryMonitor:
    """Monitor and log memory usage on Jetson devices"""
    
    def __init__(self, log_file=None, enable_logging=True):
        """
        Args:
            log_file: Path to log file (optional)
            enable_logging: Enable detailed logging
        """
        self.enable_logging = enable_logging
        self.log_file = Path(log_file) if log_file else None
        self.peak_memory = 0
        self.measurements = []
        
        # Check if we're on Jetson
        self.is_jetson = self._check_jetson()
        
        if enable_logging:
            print(f"💾 Memory Monitor initialized")
            print(f"   Platform: {'Jetson' if self.is_jetson else 'Generic'}")
    
    def _check_jetson(self):
        """Check if running on Jetson device"""
        jetson_files = [
            '/etc/nv_tegra_release',
            '/sys/module/tegra_fuse',
        ]
        return any(Path(f).exists() for f in jetson_files)
    
    def get_cpu_memory(self):
        """Get current CPU memory usage in MB"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    
    def get_gpu_memory(self):
        """Get GPU memory usage in MB (Jetson-specific)"""
        if not self.is_jetson:
            return 0.0
        
        try:
            # Try to read GPU memory from sysfs
            with open('/sys/devices/gpu.0/load', 'r') as f:
                gpu_load = int(f.read().strip())
            
            # Estimate GPU memory based on load (not accurate but gives indication)
            # On Jetson, unified memory is used
            return self.get_cpu_memory() * (gpu_load / 1000.0)
        except:
            return 0.0
    
    def get_system_memory(self):
        """Get total system memory statistics"""
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / (1024 * 1024),
            'available_mb': mem.available / (1024 * 1024),
            'used_mb': mem.used / (1024 * 1024),
            'percent': mem.percent
        }
    
    def measure(self, label=""):
        """Take a memory measurement"""
        cpu_mem = self.get_cpu_memory()
        gpu_mem = self.get_gpu_memory()
        sys_mem = self.get_system_memory()
        
        measurement = {
            'label': label,
            'cpu_mb': cpu_mem,
            'gpu_mb': gpu_mem,
            'system_used_mb': sys_mem['used_mb'],
            'system_percent': sys_mem['percent']
        }
        
        self.measurements.append(measurement)
        
        # Track peak
        if cpu_mem > self.peak_memory:
            self.peak_memory = cpu_mem
        
        if self.enable_logging:
            print(f"📊 {label if label else 'Memory'}: "
                  f"Process={cpu_mem:.1f}MB, "
                  f"System={sys_mem['used_mb']:.0f}MB ({sys_mem['percent']:.1f}%)")
        
        # Write to log file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{label},{cpu_mem:.2f},{gpu_mem:.2f},"
                       f"{sys_mem['used_mb']:.2f},{sys_mem['percent']:.2f}\n")
        
        return measurement
    
    def get_peak_memory(self):
        """Get peak memory usage"""
        return self.peak_memory
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        if self.enable_logging:
            print("🧹 Forced memory cleanup")
    
    def get_summary(self):
        """Get summary of memory measurements"""
        if not self.measurements:
            return "No measurements taken"
        
        cpu_mems = [m['cpu_mb'] for m in self.measurements]
        sys_percents = [m['system_percent'] for m in self.measurements]
        
        summary = {
            'num_measurements': len(self.measurements),
            'avg_cpu_mb': sum(cpu_mems) / len(cpu_mems),
            'peak_cpu_mb': max(cpu_mems),
            'min_cpu_mb': min(cpu_mems),
            'avg_system_percent': sum(sys_percents) / len(sys_percents),
            'peak_system_percent': max(sys_percents)
        }
        
        return summary
    
    def print_summary(self):
        """Print memory usage summary"""
        summary = self.get_summary()
        
        if isinstance(summary, str):
            print(summary)
            return
        
        print("\n" + "="*50)
        print("📊 MEMORY USAGE SUMMARY")
        print("="*50)
        print(f"Measurements taken: {summary['num_measurements']}")
        print(f"\nProcess Memory:")
        print(f"  Average: {summary['avg_cpu_mb']:.1f} MB")
        print(f"  Peak:    {summary['peak_cpu_mb']:.1f} MB")
        print(f"  Min:     {summary['min_cpu_mb']:.1f} MB")
        print(f"\nSystem Memory:")
        print(f"  Average: {summary['avg_system_percent']:.1f}%")
        print(f"  Peak:    {summary['peak_system_percent']:.1f}%")
        print("="*50 + "\n")
    
    def check_memory_threshold(self, threshold_mb=3000, warning=True):
        """
        Check if memory usage exceeds threshold
        
        Args:
            threshold_mb: Memory threshold in MB (default 3GB for Jetson Nano 4GB)
            warning: Print warning if exceeded
            
        Returns:
            True if under threshold, False otherwise
        """
        current_mem = self.get_cpu_memory()
        sys_mem = self.get_system_memory()
        
        if sys_mem['used_mb'] > threshold_mb:
            if warning:
                print(f"⚠️  WARNING: System memory usage ({sys_mem['used_mb']:.0f}MB) "
                      f"exceeds threshold ({threshold_mb}MB)")
            return False
        
        return True


# Convenience function
def measure_memory(func):
    """Decorator to measure memory usage of a function"""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor(enable_logging=True)
        monitor.measure("Before function")
        result = func(*args, **kwargs)
        monitor.measure("After function")
        monitor.print_summary()
        return result
    return wrapper


if __name__ == "__main__":
    """Test memory monitoring"""
    print("Testing Memory Monitor...")
    
    monitor = MemoryMonitor(enable_logging=True)
    
    # Take initial measurement
    monitor.measure("Startup")
    
    # Simulate some work
    import numpy as np
    import time
    
    print("\nAllocating arrays...")
    arrays = []
    for i in range(5):
        # Allocate 100MB array
        arr = np.random.rand(1000, 1000, 10)
        arrays.append(arr)
        monitor.measure(f"After array {i+1}")
        time.sleep(0.5)
    
    print("\nCleaning up...")
    del arrays
    monitor.force_cleanup()
    monitor.measure("After cleanup")
    
    # Print summary
    monitor.print_summary()
    
    # Check threshold
    monitor.check_memory_threshold(threshold_mb=3000)
