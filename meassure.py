import time
import psutil
import os
import threading
import json
from pathlib import Path

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

PROFILE_LOG = Path("profiling_log.json")

def profile_resources(func):
    """
    Decorator to measure:
      - Execution time
      - Average and peak RAM usage
      - Average and peak CPU usage (normalized 0–100%)
      - Optional GPU memory usage (if PyTorch is available)
    Logs results to profiling_log.json
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        ram_samples = []
        cpu_samples = []
        cpu_count = psutil.cpu_count(logical=True)

        # Prime cpu_percent to avoid bogus first value
        process.cpu_percent(interval=None)

        # Monitoring thread
        def monitor():
            while not stop_monitoring.is_set():
                ram_samples.append(process.memory_info().rss)

                # use a short but real interval for accuracy
                cpu_raw = process.cpu_percent(interval=0.05)
                cpu_normalized = min(cpu_raw / cpu_count, 100.0)  # clamp to 100%
                cpu_samples.append(cpu_normalized)

        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        # Time the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()

        # Compute stats
        exec_time = end_time - start_time
        avg_ram = sum(ram_samples)/len(ram_samples)/1024/1024 if ram_samples else 0
        peak_ram = max(ram_samples)/1024/1024 if ram_samples else 0
        avg_cpu = sum(cpu_samples)/len(cpu_samples) if cpu_samples else 0
        peak_cpu = max(cpu_samples) if cpu_samples else 0
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2 if GPU_AVAILABLE else None

        # Prepare log entry
        log_entry = {
            "function": func.__name__,
            "execution_time_s": exec_time,
            "ram_avg_mb": avg_ram,
            "ram_peak_mb": peak_ram,
            "cpu_avg_percent": avg_cpu,
            "cpu_peak_percent": peak_cpu,
        }
        if GPU_AVAILABLE:
            log_entry["gpu_peak_mb"] = gpu_peak

        # Save to JSON file
        if PROFILE_LOG.exists():
            with open(PROFILE_LOG, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(log_entry)
        with open(PROFILE_LOG, "w") as f:
            json.dump(data, f, indent=2)

        # Print summary
        print(f"{func.__name__} execution time: {exec_time:.4f} s")
        print(f"{func.__name__} RAM usage - Avg: {avg_ram:.2f} MB, Peak: {peak_ram:.2f} MB")
        print(f"{func.__name__} CPU usage - Avg: {avg_cpu:.2f}%, Peak: {peak_cpu:.2f}%")
        if GPU_AVAILABLE:
            print(f"{func.__name__} GPU memory peak: {gpu_peak:.2f} MB")

        return result

    return wrapper
