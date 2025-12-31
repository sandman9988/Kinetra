"""
Hardware Optimizer Module
=========================

Auto-detects hardware capabilities and optimizes system configuration:
- CPU detection (cores, cache, frequency, architecture)
- Memory detection (RAM, swap, available)
- GPU detection (CUDA, OpenCL, memory)
- Disk I/O profiling
- Network interface detection
- Auto-tuning based on hardware profile

Optimization targets:
- Thread/process pool sizing
- Buffer sizes (ring buffers, caches)
- Batch sizes for processing
- Memory allocation strategies
- GPU offloading decisions
"""

import gc
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CPUArchitecture(Enum):
    """CPU architecture types."""
    X86_64 = auto()
    ARM64 = auto()
    ARM32 = auto()
    X86 = auto()
    UNKNOWN = auto()


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    APPLE = auto()  # Apple Silicon
    NONE = auto()


class PerformanceTier(Enum):
    """System performance tier classification."""
    ULTRA = auto()      # High-end workstation/server
    HIGH = auto()       # Gaming PC / workstation
    MEDIUM = auto()     # Standard desktop
    LOW = auto()        # Laptop / older hardware
    MINIMAL = auto()    # Embedded / very limited


@dataclass
class CPUInfo:
    """CPU information and capabilities."""
    model: str = "Unknown"
    architecture: CPUArchitecture = CPUArchitecture.UNKNOWN
    physical_cores: int = 1
    logical_cores: int = 1
    frequency_mhz: float = 0.0
    frequency_max_mhz: float = 0.0
    cache_l1_kb: int = 0
    cache_l2_kb: int = 0
    cache_l3_kb: int = 0
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_sse4: bool = False
    has_neon: bool = False  # ARM SIMD
    numa_nodes: int = 1
    
    @property
    def total_cache_kb(self) -> int:
        return self.cache_l1_kb + self.cache_l2_kb + self.cache_l3_kb
    
    @property
    def simd_width(self) -> int:
        """Best SIMD width in bits."""
        if self.has_avx512:
            return 512
        if self.has_avx2 or self.has_avx:
            return 256
        if self.has_sse4:
            return 128
        if self.has_neon:
            return 128
        return 64


@dataclass
class MemoryInfo:
    """Memory information and capabilities."""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    total_swap_gb: float = 0.0
    available_swap_gb: float = 0.0
    memory_speed_mhz: int = 0
    memory_channels: int = 1
    page_size_kb: int = 4
    huge_pages_available: bool = False
    huge_page_size_kb: int = 2048
    
    @property
    def usable_ram_gb(self) -> float:
        """RAM available for application use (80% of available)."""
        return self.available_ram_gb * 0.8


@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    name: str = "None"
    vendor: GPUVendor = GPUVendor.NONE
    memory_gb: float = 0.0
    memory_available_gb: float = 0.0
    compute_capability: str = ""
    cuda_cores: int = 0
    tensor_cores: int = 0
    clock_mhz: int = 0
    driver_version: str = ""
    cuda_available: bool = False
    opencl_available: bool = False
    metal_available: bool = False  # Apple
    
    @property
    def is_available(self) -> bool:
        return self.vendor != GPUVendor.NONE


@dataclass
class DiskInfo:
    """Disk I/O information."""
    total_gb: float = 0.0
    available_gb: float = 0.0
    is_ssd: bool = False
    is_nvme: bool = False
    read_speed_mbps: float = 0.0
    write_speed_mbps: float = 0.0
    iops_read: int = 0
    iops_write: int = 0


@dataclass
class NetworkInfo:
    """Network interface information."""
    interfaces: List[str] = field(default_factory=list)
    max_bandwidth_mbps: float = 0.0
    has_10gbe: bool = False
    has_infiniband: bool = False


@dataclass
class HardwareProfile:
    """Complete hardware profile."""
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    disk: DiskInfo = field(default_factory=DiskInfo)
    network: NetworkInfo = field(default_factory=NetworkInfo)
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    tier: PerformanceTier = PerformanceTier.MEDIUM
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cpu': {
                'model': self.cpu.model,
                'architecture': self.cpu.architecture.name,
                'physical_cores': self.cpu.physical_cores,
                'logical_cores': self.cpu.logical_cores,
                'frequency_mhz': self.cpu.frequency_mhz,
                'cache_total_kb': self.cpu.total_cache_kb,
                'simd_width': self.cpu.simd_width,
            },
            'memory': {
                'total_gb': self.memory.total_ram_gb,
                'available_gb': self.memory.available_ram_gb,
                'usable_gb': self.memory.usable_ram_gb,
            },
            'gpu': {
                'name': self.gpu.name,
                'vendor': self.gpu.vendor.name,
                'memory_gb': self.gpu.memory_gb,
                'cuda_available': self.gpu.cuda_available,
            },
            'disk': {
                'total_gb': self.disk.total_gb,
                'available_gb': self.disk.available_gb,
                'is_ssd': self.disk.is_ssd,
            },
            'tier': self.tier.name,
            'os': f"{self.os_name} {self.os_version}",
            'python': self.python_version,
        }


class HardwareDetector:
    """
    Detect hardware capabilities.
    
    Supports Linux, macOS, and Windows.
    """
    
    def __init__(self):
        """Initialize hardware detector."""
        self._profile: Optional[HardwareProfile] = None
        self._detection_time: Optional[float] = None
    
    def detect(self, force: bool = False) -> HardwareProfile:
        """
        Detect hardware profile.
        
        Args:
            force: Force re-detection even if cached
            
        Returns:
            Hardware profile
        """
        if self._profile and not force:
            return self._profile
        
        start = time.time()
        
        profile = HardwareProfile(
            cpu=self._detect_cpu(),
            memory=self._detect_memory(),
            gpu=self._detect_gpu(),
            disk=self._detect_disk(),
            network=self._detect_network(),
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
        )
        
        # Determine performance tier
        profile.tier = self._classify_tier(profile)
        
        self._profile = profile
        self._detection_time = time.time() - start
        
        logger.info(f"Hardware detection completed in {self._detection_time:.2f}s")
        logger.info(f"Performance tier: {profile.tier.name}")
        
        return profile
    
    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU information."""
        info = CPUInfo()
        
        # Basic info from Python
        info.physical_cores = os.cpu_count() or 1
        info.logical_cores = os.cpu_count() or 1
        
        # Architecture
        machine = platform.machine().lower()
        if 'x86_64' in machine or 'amd64' in machine:
            info.architecture = CPUArchitecture.X86_64
        elif 'aarch64' in machine or 'arm64' in machine:
            info.architecture = CPUArchitecture.ARM64
            info.has_neon = True
        elif 'arm' in machine:
            info.architecture = CPUArchitecture.ARM32
            info.has_neon = True
        elif 'i386' in machine or 'i686' in machine:
            info.architecture = CPUArchitecture.X86
        
        # Try to get more detailed info
        system = platform.system()
        
        if system == 'Linux':
            info = self._detect_cpu_linux(info)
        elif system == 'Darwin':
            info = self._detect_cpu_macos(info)
        elif system == 'Windows':
            info = self._detect_cpu_windows(info)
        
        return info
    
    def _detect_cpu_linux(self, info: CPUInfo) -> CPUInfo:
        """Detect CPU info on Linux."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Model name
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    info.model = line.split(':')[1].strip()
                    break
            
            # Physical cores
            physical = set()
            for line in cpuinfo.split('\n'):
                if 'physical id' in line:
                    physical.add(line.split(':')[1].strip())
            
            if physical:
                cores_per_socket = cpuinfo.count('processor')
                info.physical_cores = len(physical) * (cores_per_socket // max(len(physical), 1))
            
            # CPU flags for SIMD
            for line in cpuinfo.split('\n'):
                if 'flags' in line:
                    flags = line.lower()
                    info.has_avx = 'avx' in flags
                    info.has_avx2 = 'avx2' in flags
                    info.has_avx512 = 'avx512' in flags
                    info.has_sse4 = 'sse4' in flags
                    break
            
            # Frequency
            try:
                with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq', 'r') as f:
                    info.frequency_max_mhz = int(f.read().strip()) / 1000
            except (FileNotFoundError, PermissionError):
                pass
            
            # Cache sizes
            try:
                cache_path = Path('/sys/devices/system/cpu/cpu0/cache')
                for index_dir in cache_path.glob('index*'):
                    level_file = index_dir / 'level'
                    size_file = index_dir / 'size'
                    
                    if level_file.exists() and size_file.exists():
                        level = int(level_file.read_text().strip())
                        size_str = size_file.read_text().strip()
                        size_kb = int(size_str.replace('K', ''))
                        
                        if level == 1:
                            info.cache_l1_kb += size_kb
                        elif level == 2:
                            info.cache_l2_kb = size_kb
                        elif level == 3:
                            info.cache_l3_kb = size_kb
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            
        except Exception as e:
            logger.warning(f"Linux CPU detection error: {e}")
        
        return info
    
    def _detect_cpu_macos(self, info: CPUInfo) -> CPUInfo:
        """Detect CPU info on macOS."""
        try:
            # Use sysctl
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True, text=True, timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if 'machdep.cpu.brand_string' in line:
                    info.model = line.split(':')[1].strip()
                elif 'hw.physicalcpu' in line and 'max' not in line:
                    info.physical_cores = int(line.split(':')[1].strip())
                elif 'hw.logicalcpu' in line and 'max' not in line:
                    info.logical_cores = int(line.split(':')[1].strip())
                elif 'hw.cpufrequency_max' in line:
                    info.frequency_max_mhz = int(line.split(':')[1].strip()) / 1_000_000
                elif 'hw.l2cachesize' in line:
                    info.cache_l2_kb = int(line.split(':')[1].strip()) // 1024
                elif 'hw.l3cachesize' in line:
                    info.cache_l3_kb = int(line.split(':')[1].strip()) // 1024
                elif 'machdep.cpu.features' in line:
                    features = line.lower()
                    info.has_avx = 'avx' in features
                    info.has_avx2 = 'avx2' in features
                    info.has_sse4 = 'sse4' in features
                    
        except Exception as e:
            logger.warning(f"macOS CPU detection error: {e}")
        
        return info
    
    def _detect_cpu_windows(self, info: CPUInfo) -> CPUInfo:
        """Detect CPU info on Windows."""
        try:
            import winreg
            
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            
            info.model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            info.frequency_mhz = winreg.QueryValueEx(key, "~MHz")[0]
            
            winreg.CloseKey(key)
            
        except Exception as e:
            logger.warning(f"Windows CPU detection error: {e}")
        
        return info
    
    def _detect_memory(self) -> MemoryInfo:
        """Detect memory information."""
        info = MemoryInfo()
        
        system = platform.system()
        
        if system == 'Linux':
            info = self._detect_memory_linux(info)
        elif system == 'Darwin':
            info = self._detect_memory_macos(info)
        elif system == 'Windows':
            info = self._detect_memory_windows(info)
        
        # Fallback using Python
        if info.total_ram_gb == 0:
            try:
                import psutil
                vm = psutil.virtual_memory()
                info.total_ram_gb = vm.total / (1024**3)
                info.available_ram_gb = vm.available / (1024**3)
            except ImportError:
                # Very rough estimate
                info.total_ram_gb = 8.0
                info.available_ram_gb = 4.0
        
        return info
    
    def _detect_memory_linux(self, info: MemoryInfo) -> MemoryInfo:
        """Detect memory on Linux."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    kb = int(line.split()[1])
                    info.total_ram_gb = kb / (1024**2)
                elif 'MemAvailable' in line:
                    kb = int(line.split()[1])
                    info.available_ram_gb = kb / (1024**2)
                elif 'SwapTotal' in line:
                    kb = int(line.split()[1])
                    info.total_swap_gb = kb / (1024**2)
                elif 'SwapFree' in line:
                    kb = int(line.split()[1])
                    info.available_swap_gb = kb / (1024**2)
                elif 'Hugepagesize' in line:
                    info.huge_page_size_kb = int(line.split()[1])
                    info.huge_pages_available = True
                    
        except Exception as e:
            logger.warning(f"Linux memory detection error: {e}")
        
        return info
    
    def _detect_memory_macos(self, info: MemoryInfo) -> MemoryInfo:
        """Detect memory on macOS."""
        try:
            result = subprocess.run(
                ['sysctl', 'hw.memsize'],
                capture_output=True, text=True, timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if 'hw.memsize' in line:
                    bytes_val = int(line.split(':')[1].strip())
                    info.total_ram_gb = bytes_val / (1024**3)
            
            # Get available memory using vm_stat
            result = subprocess.run(
                ['vm_stat'],
                capture_output=True, text=True, timeout=5
            )
            
            page_size = 4096  # Default
            free_pages = 0
            inactive_pages = 0
            
            for line in result.stdout.split('\n'):
                if 'page size' in line.lower():
                    page_size = int(line.split()[-2])
                elif 'Pages free' in line:
                    free_pages = int(line.split()[-1].replace('.', ''))
                elif 'Pages inactive' in line:
                    inactive_pages = int(line.split()[-1].replace('.', ''))
            
            info.available_ram_gb = (free_pages + inactive_pages) * page_size / (1024**3)
            
        except Exception as e:
            logger.warning(f"macOS memory detection error: {e}")
        
        return info
    
    def _detect_memory_windows(self, info: MemoryInfo) -> MemoryInfo:
        """Detect memory on Windows."""
        try:
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            
            info.total_ram_gb = mem.ullTotalPhys / (1024**3)
            info.available_ram_gb = mem.ullAvailPhys / (1024**3)
            info.total_swap_gb = mem.ullTotalPageFile / (1024**3)
            info.available_swap_gb = mem.ullAvailPageFile / (1024**3)
            
        except Exception as e:
            logger.warning(f"Windows memory detection error: {e}")
        
        return info
    
    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU information."""
        info = GPUInfo()
        
        # Try NVIDIA CUDA
        info = self._detect_nvidia_gpu(info)
        
        if not info.is_available:
            # Try AMD ROCm
            info = self._detect_amd_gpu(info)
        
        if not info.is_available:
            # Try Apple Metal
            info = self._detect_apple_gpu(info)
        
        if not info.is_available:
            # Try Intel
            info = self._detect_intel_gpu(info)
        
        return info
    
    def _detect_nvidia_gpu(self, info: GPUInfo) -> GPUInfo:
        """Detect NVIDIA GPU."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,clocks.max.sm',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    info.name = parts[0].strip()
                    info.vendor = GPUVendor.NVIDIA
                    info.memory_gb = float(parts[1].strip()) / 1024
                    info.memory_available_gb = float(parts[2].strip()) / 1024
                    info.driver_version = parts[3].strip()
                    if len(parts) >= 5:
                        info.clock_mhz = int(float(parts[4].strip()))
                    info.cuda_available = True
                    
                    # Try to get compute capability
                    try:
                        import torch
                        if torch.cuda.is_available():
                            cap = torch.cuda.get_device_capability(0)
                            info.compute_capability = f"{cap[0]}.{cap[1]}"
                    except ImportError:
                        pass
                        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection error: {e}")
        
        return info
    
    def _detect_amd_gpu(self, info: GPUInfo) -> GPUInfo:
        """Detect AMD GPU."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and 'GPU' in result.stdout:
                info.vendor = GPUVendor.AMD
                for line in result.stdout.split('\n'):
                    if 'Card series' in line or 'GPU' in line:
                        info.name = line.split(':')[-1].strip()
                        break
                        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.warning(f"AMD GPU detection error: {e}")
        
        return info
    
    def _detect_apple_gpu(self, info: GPUInfo) -> GPUInfo:
        """Detect Apple GPU (Metal)."""
        if platform.system() != 'Darwin':
            return info
        
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=10
            )
            
            if 'Apple' in result.stdout or 'M1' in result.stdout or 'M2' in result.stdout:
                info.vendor = GPUVendor.APPLE
                info.metal_available = True
                
                for line in result.stdout.split('\n'):
                    if 'Chipset Model' in line:
                        info.name = line.split(':')[1].strip()
                    elif 'VRAM' in line:
                        vram_str = line.split(':')[1].strip()
                        if 'GB' in vram_str:
                            info.memory_gb = float(vram_str.replace('GB', '').strip())
                            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.warning(f"Apple GPU detection error: {e}")
        
        return info
    
    def _detect_intel_gpu(self, info: GPUInfo) -> GPUInfo:
        """Detect Intel GPU."""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True, text=True, timeout=10
                )
                
                for line in result.stdout.split('\n'):
                    if 'VGA' in line and 'Intel' in line:
                        info.vendor = GPUVendor.INTEL
                        info.name = line.split(':')[-1].strip()
                        info.opencl_available = True
                        break
                        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.warning(f"Intel GPU detection error: {e}")
        
        return info
    
    def _detect_disk(self) -> DiskInfo:
        """Detect disk information."""
        info = DiskInfo()
        
        try:
            # Get disk usage
            usage = shutil.disk_usage('/')
            info.total_gb = usage.total / (1024**3)
            info.available_gb = usage.free / (1024**3)
            
            # Detect SSD/NVMe on Linux
            if platform.system() == 'Linux':
                try:
                    # Check if root is on SSD
                    result = subprocess.run(
                        ['lsblk', '-d', '-o', 'name,rota'],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    for line in result.stdout.split('\n'):
                        if 'sd' in line or 'nvme' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                if 'nvme' in parts[0]:
                                    info.is_nvme = True
                                    info.is_ssd = True
                                elif parts[1] == '0':
                                    info.is_ssd = True
                            break
                            
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                    
        except Exception as e:
            logger.warning(f"Disk detection error: {e}")
        
        return info
    
    def _detect_network(self) -> NetworkInfo:
        """Detect network information."""
        info = NetworkInfo()
        
        try:
            if platform.system() == 'Linux':
                # List network interfaces
                net_path = Path('/sys/class/net')
                if net_path.exists():
                    for iface in net_path.iterdir():
                        if iface.name != 'lo':
                            info.interfaces.append(iface.name)
                            
                            # Check speed
                            speed_file = iface / 'speed'
                            if speed_file.exists():
                                try:
                                    speed = int(speed_file.read_text().strip())
                                    if speed > info.max_bandwidth_mbps:
                                        info.max_bandwidth_mbps = speed
                                    if speed >= 10000:
                                        info.has_10gbe = True
                                except (ValueError, PermissionError):
                                    pass
                                    
        except Exception as e:
            logger.warning(f"Network detection error: {e}")
        
        return info
    
    def _classify_tier(self, profile: HardwareProfile) -> PerformanceTier:
        """Classify system into performance tier."""
        score = 0
        
        # CPU score (0-40)
        cpu_score = min(40, profile.cpu.logical_cores * 2)
        if profile.cpu.has_avx2:
            cpu_score += 5
        if profile.cpu.has_avx512:
            cpu_score += 5
        score += cpu_score
        
        # Memory score (0-30)
        mem_score = min(30, profile.memory.total_ram_gb * 2)
        score += mem_score
        
        # GPU score (0-30)
        if profile.gpu.is_available:
            gpu_score = min(30, profile.gpu.memory_gb * 3)
            if profile.gpu.cuda_available:
                gpu_score += 10
            score += gpu_score
        
        # Classify
        if score >= 80:
            return PerformanceTier.ULTRA
        elif score >= 60:
            return PerformanceTier.HIGH
        elif score >= 40:
            return PerformanceTier.MEDIUM
        elif score >= 20:
            return PerformanceTier.LOW
        else:
            return PerformanceTier.MINIMAL
    
    @property
    def profile(self) -> HardwareProfile:
        """Get cached profile or detect."""
        if not self._profile:
            self.detect()
        return self._profile


@dataclass
class OptimizedConfig:
    """Optimized configuration based on hardware."""
    # Thread/process pools
    cpu_workers: int = 4
    io_workers: int = 8
    use_processes: bool = False  # vs threads
    
    # Buffer sizes
    tick_buffer_size: int = 10000
    bar_buffer_size: int = 1000
    cache_size: int = 1000
    
    # Batch sizes
    tick_batch_size: int = 10
    indicator_batch_size: int = 100
    backtest_batch_size: int = 1000
    
    # Memory management
    max_memory_gb: float = 4.0
    use_memory_mapping: bool = False
    preallocate_buffers: bool = True
    
    # GPU settings
    use_gpu: bool = False
    gpu_batch_size: int = 1024
    gpu_memory_fraction: float = 0.8
    
    # Network settings
    connection_pool_size: int = 10
    socket_buffer_size: int = 65536
    
    # Misc
    use_simd: bool = True
    use_jit: bool = False  # Numba JIT
    enable_profiling: bool = False


class HardwareOptimizer:
    """
    Optimize system configuration based on hardware.
    
    Provides hardware-aware tuning for:
    - Thread/process pool sizing
    - Buffer and cache sizes
    - Batch sizes for operations
    - Memory allocation
    - GPU utilization
    """
    
    def __init__(self, detector: HardwareDetector = None):
        """
        Initialize optimizer.
        
        Args:
            detector: Hardware detector (auto-creates if None)
        """
        self._detector = detector or HardwareDetector()
        self._config: Optional[OptimizedConfig] = None
    
    def optimize(self, workload: str = 'balanced') -> OptimizedConfig:
        """
        Generate optimized configuration.
        
        Args:
            workload: Workload type ('balanced', 'latency', 'throughput', 'memory')
            
        Returns:
            Optimized configuration
        """
        profile = self._detector.detect()
        config = OptimizedConfig()
        
        # Base optimization by tier
        if profile.tier == PerformanceTier.ULTRA:
            config = self._optimize_ultra(profile, config)
        elif profile.tier == PerformanceTier.HIGH:
            config = self._optimize_high(profile, config)
        elif profile.tier == PerformanceTier.MEDIUM:
            config = self._optimize_medium(profile, config)
        elif profile.tier == PerformanceTier.LOW:
            config = self._optimize_low(profile, config)
        else:
            config = self._optimize_minimal(profile, config)
        
        # Workload-specific adjustments
        if workload == 'latency':
            config = self._adjust_for_latency(config)
        elif workload == 'throughput':
            config = self._adjust_for_throughput(config)
        elif workload == 'memory':
            config = self._adjust_for_memory(config)
        
        self._config = config
        return config
    
    def _optimize_ultra(self, profile: HardwareProfile, config: OptimizedConfig) -> OptimizedConfig:
        """Optimize for ultra-high-end hardware."""
        cpu = profile.cpu
        mem = profile.memory
        gpu = profile.gpu
        
        # Use all cores minus 2 for system
        config.cpu_workers = max(1, cpu.physical_cores - 2)
        config.io_workers = cpu.logical_cores
        config.use_processes = True  # Can afford process overhead
        
        # Large buffers
        config.tick_buffer_size = 100000
        config.bar_buffer_size = 10000
        config.cache_size = 10000
        
        # Large batches
        config.tick_batch_size = 100
        config.indicator_batch_size = 1000
        config.backtest_batch_size = 10000
        
        # Memory
        config.max_memory_gb = min(mem.usable_ram_gb, 32.0)
        config.use_memory_mapping = True
        config.preallocate_buffers = True
        
        # GPU
        if gpu.is_available and gpu.cuda_available:
            config.use_gpu = True
            config.gpu_batch_size = 4096
            config.gpu_memory_fraction = 0.8
        
        # Network
        config.connection_pool_size = 50
        config.socket_buffer_size = 262144
        
        # Advanced
        config.use_simd = cpu.simd_width >= 256
        config.use_jit = True
        
        return config
    
    def _optimize_high(self, profile: HardwareProfile, config: OptimizedConfig) -> OptimizedConfig:
        """Optimize for high-end hardware."""
        cpu = profile.cpu
        mem = profile.memory
        gpu = profile.gpu
        
        config.cpu_workers = max(1, cpu.physical_cores - 1)
        config.io_workers = min(cpu.logical_cores, 16)
        config.use_processes = cpu.physical_cores >= 8
        
        config.tick_buffer_size = 50000
        config.bar_buffer_size = 5000
        config.cache_size = 5000
        
        config.tick_batch_size = 50
        config.indicator_batch_size = 500
        config.backtest_batch_size = 5000
        
        config.max_memory_gb = min(mem.usable_ram_gb, 16.0)
        config.use_memory_mapping = mem.total_ram_gb >= 16
        config.preallocate_buffers = True
        
        if gpu.is_available and gpu.cuda_available:
            config.use_gpu = True
            config.gpu_batch_size = 2048
            config.gpu_memory_fraction = 0.7
        
        config.connection_pool_size = 20
        config.socket_buffer_size = 131072
        
        config.use_simd = cpu.simd_width >= 128
        config.use_jit = cpu.physical_cores >= 6
        
        return config
    
    def _optimize_medium(self, profile: HardwareProfile, config: OptimizedConfig) -> OptimizedConfig:
        """Optimize for medium hardware."""
        cpu = profile.cpu
        mem = profile.memory
        
        config.cpu_workers = max(1, cpu.physical_cores // 2)
        config.io_workers = min(cpu.logical_cores, 8)
        config.use_processes = False  # Threads more efficient
        
        config.tick_buffer_size = 10000
        config.bar_buffer_size = 1000
        config.cache_size = 1000
        
        config.tick_batch_size = 10
        config.indicator_batch_size = 100
        config.backtest_batch_size = 1000
        
        config.max_memory_gb = min(mem.usable_ram_gb, 4.0)
        config.use_memory_mapping = False
        config.preallocate_buffers = True
        
        config.use_gpu = False
        
        config.connection_pool_size = 10
        config.socket_buffer_size = 65536
        
        config.use_simd = True
        config.use_jit = False
        
        return config
    
    def _optimize_low(self, profile: HardwareProfile, config: OptimizedConfig) -> OptimizedConfig:
        """Optimize for low-end hardware."""
        cpu = profile.cpu
        mem = profile.memory
        
        config.cpu_workers = max(1, cpu.physical_cores // 2)
        config.io_workers = min(cpu.logical_cores, 4)
        config.use_processes = False
        
        config.tick_buffer_size = 5000
        config.bar_buffer_size = 500
        config.cache_size = 500
        
        config.tick_batch_size = 5
        config.indicator_batch_size = 50
        config.backtest_batch_size = 500
        
        config.max_memory_gb = min(mem.usable_ram_gb, 2.0)
        config.use_memory_mapping = False
        config.preallocate_buffers = False  # Save memory
        
        config.use_gpu = False
        
        config.connection_pool_size = 5
        config.socket_buffer_size = 32768
        
        config.use_simd = True
        config.use_jit = False
        
        return config
    
    def _optimize_minimal(self, profile: HardwareProfile, config: OptimizedConfig) -> OptimizedConfig:
        """Optimize for minimal hardware."""
        config.cpu_workers = 1
        config.io_workers = 2
        config.use_processes = False
        
        config.tick_buffer_size = 1000
        config.bar_buffer_size = 100
        config.cache_size = 100
        
        config.tick_batch_size = 1
        config.indicator_batch_size = 10
        config.backtest_batch_size = 100
        
        config.max_memory_gb = 0.5
        config.use_memory_mapping = False
        config.preallocate_buffers = False
        
        config.use_gpu = False
        
        config.connection_pool_size = 2
        config.socket_buffer_size = 16384
        
        config.use_simd = False
        config.use_jit = False
        
        return config
    
    def _adjust_for_latency(self, config: OptimizedConfig) -> OptimizedConfig:
        """Adjust config for low latency."""
        # Smaller batches = faster response
        config.tick_batch_size = max(1, config.tick_batch_size // 2)
        config.indicator_batch_size = max(10, config.indicator_batch_size // 2)
        
        # More I/O workers
        config.io_workers = min(config.io_workers * 2, 32)
        
        # Smaller buffers = less processing delay
        config.tick_buffer_size = max(1000, config.tick_buffer_size // 2)
        
        # Preallocate to avoid allocation latency
        config.preallocate_buffers = True
        
        return config
    
    def _adjust_for_throughput(self, config: OptimizedConfig) -> OptimizedConfig:
        """Adjust config for maximum throughput."""
        # Larger batches = better throughput
        config.tick_batch_size *= 2
        config.indicator_batch_size *= 2
        config.backtest_batch_size *= 2
        
        # Larger buffers
        config.tick_buffer_size *= 2
        config.bar_buffer_size *= 2
        config.cache_size *= 2
        
        # Enable GPU if available
        if self._detector.profile.gpu.is_available:
            config.use_gpu = True
            config.gpu_batch_size *= 2
        
        return config
    
    def _adjust_for_memory(self, config: OptimizedConfig) -> OptimizedConfig:
        """Adjust config for memory efficiency."""
        # Smaller buffers
        config.tick_buffer_size = max(1000, config.tick_buffer_size // 4)
        config.bar_buffer_size = max(100, config.bar_buffer_size // 4)
        config.cache_size = max(100, config.cache_size // 4)
        
        # Reduce max memory
        config.max_memory_gb = max(0.5, config.max_memory_gb / 2)
        
        # Don't preallocate
        config.preallocate_buffers = False
        
        # Disable GPU (saves GPU memory)
        config.use_gpu = False
        
        # Use threads (less memory than processes)
        config.use_processes = False
        
        return config
    
    def apply_config(self, config: OptimizedConfig = None) -> Dict[str, Any]:
        """
        Apply configuration to system.
        
        Returns:
            Applied settings summary
        """
        config = config or self._config or self.optimize()
        
        applied = {}
        
        # Set environment variables for numpy/etc
        if config.cpu_workers:
            os.environ['OMP_NUM_THREADS'] = str(config.cpu_workers)
            os.environ['MKL_NUM_THREADS'] = str(config.cpu_workers)
            os.environ['NUMEXPR_NUM_THREADS'] = str(config.cpu_workers)
            applied['thread_env'] = config.cpu_workers
        
        # Configure numpy
        try:
            np.set_printoptions(precision=8, suppress=True)
            applied['numpy_configured'] = True
        except Exception:
            pass
        
        # GPU configuration
        if config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
                    applied['gpu_memory_fraction'] = config.gpu_memory_fraction
            except ImportError:
                pass
        
        # Garbage collection tuning
        if config.preallocate_buffers:
            gc.set_threshold(700, 10, 10)  # Less aggressive GC
            applied['gc_threshold'] = (700, 10, 10)
        
        logger.info(f"Applied hardware-optimized configuration: {applied}")
        
        return applied
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        profile = self._detector.detect()
        recommendations = []
        
        # CPU recommendations
        if profile.cpu.physical_cores < 4:
            recommendations.append(
                "Consider reducing concurrent operations - limited CPU cores detected"
            )
        
        if not profile.cpu.has_avx2:
            recommendations.append(
                "CPU lacks AVX2 - some vectorized operations may be slower"
            )
        
        # Memory recommendations
        if profile.memory.total_ram_gb < 8:
            recommendations.append(
                "Low RAM detected - consider reducing buffer sizes and using streaming"
            )
        
        if profile.memory.available_ram_gb < 2:
            recommendations.append(
                "WARNING: Very low available RAM - system may experience swapping"
            )
        
        # GPU recommendations
        if profile.gpu.is_available and profile.gpu.cuda_available:
            recommendations.append(
                f"CUDA GPU detected ({profile.gpu.name}) - enable GPU acceleration for speedup"
            )
        elif profile.gpu.vendor == GPUVendor.AMD:
            recommendations.append(
                "AMD GPU detected - consider using OpenCL or ROCm for acceleration"
            )
        
        # Disk recommendations
        if not profile.disk.is_ssd:
            recommendations.append(
                "HDD detected - consider using SSD for faster data loading"
            )
        
        if profile.disk.available_gb < 10:
            recommendations.append(
                "Low disk space - clean up to avoid I/O issues"
            )
        
        # Network recommendations
        if profile.network.max_bandwidth_mbps < 100:
            recommendations.append(
                "Slow network detected - consider latency-optimized configuration"
            )
        
        return recommendations
    
    @property
    def config(self) -> OptimizedConfig:
        """Get current config or generate."""
        if not self._config:
            self.optimize()
        return self._config


def auto_configure() -> Tuple[HardwareProfile, OptimizedConfig]:
    """
    Auto-detect hardware and generate optimized config.
    
    Returns:
        (HardwareProfile, OptimizedConfig)
    """
    detector = HardwareDetector()
    profile = detector.detect()
    
    optimizer = HardwareOptimizer(detector)
    config = optimizer.optimize()
    optimizer.apply_config(config)
    
    return profile, config


# Export all components
__all__ = [
    'CPUArchitecture',
    'GPUVendor',
    'PerformanceTier',
    'CPUInfo',
    'MemoryInfo',
    'GPUInfo',
    'DiskInfo',
    'NetworkInfo',
    'HardwareProfile',
    'HardwareDetector',
    'OptimizedConfig',
    'HardwareOptimizer',
    'auto_configure',
]
