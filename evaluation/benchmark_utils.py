import os
import glob
import time
import threading
from typing import Optional
import psutil
import pynvml

def find_first_mrc_file(tomogram_dir: str) -> Optional[str]:
    pattern = os.path.join(tomogram_dir, "*.mrc")
    mrc_files = sorted(glob.glob(pattern))
    if not mrc_files:
        return None
    return mrc_files[0]


class CPUMonitor:
    def __init__(self, interval: float = 0.1, include_children: bool = True):
        self.interval = interval
        self.include_children = include_children
        self.process_pid: Optional[int] = None
        self.max_rss_mb: float = 0.0
        self.monitoring: bool = False
        self.monitor_thread: Optional[threading.Thread] = None

    def _get_rss_mb(self, pid: int) -> float:
        try:
            proc = psutil.Process(pid)
        except Exception:
            return 0.0

        total_rss = 0
        try:
            total_rss += proc.memory_info().rss
        except Exception:
            pass

        if self.include_children:
            try:
                for child in proc.children(recursive=True):
                    try:
                        total_rss += child.memory_info().rss
                    except Exception:
                        continue
            except Exception:
                pass

        return total_rss / (1024 ** 2)

    def _monitor_loop(self):
        while self.monitoring and self.process_pid is not None:
            try:
                rss_mb = self._get_rss_mb(self.process_pid)
                if rss_mb > self.max_rss_mb:
                    self.max_rss_mb = rss_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self, process_pid: int):
        self.process_pid = process_pid
        self.max_rss_mb = 0.0
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> float:
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.max_rss_mb


class GPUMonitor:
    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.device_id = device_id
        self.interval = interval
        self.process_pid: Optional[int] = None
        self.max_memory_mb: float = 0.0
        self.monitoring: bool = False
        self.monitor_thread: Optional[threading.Thread] = None

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def _get_process_memory_mb(self, pid: int) -> float:
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            for proc in processes:
                if proc.pid == pid:
                    return proc.usedGpuMemory / (1024 ** 2)  # bytes -> MB
            return 0.0
        except Exception:
            return 0.0

    def _monitor_loop(self):
        while self.monitoring and self.process_pid is not None:
            try:
                memory_mb = self._get_process_memory_mb(self.process_pid)
                if memory_mb > self.max_memory_mb:
                    self.max_memory_mb = memory_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self, process_pid: int):
        self.process_pid = process_pid
        self.max_memory_mb = 0.0
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> float:
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.max_memory_mb

