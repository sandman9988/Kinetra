"""
GPU Auto-Detection and Optimization
====================================

Automatically detect and configure GPU acceleration:
- NVIDIA CUDA
- AMD ROCm
- Apple Metal (MPS)
- CPU fallback
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class GPUType(Enum):
    """Supported GPU types."""
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    APPLE_MPS = "apple_mps"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    gpu_type: GPUType
    name: str
    memory_total_mb: int
    memory_free_mb: int
    compute_capability: str
    driver_version: str
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    device_index: int = 0


class GPUManager:
    """
    Manages GPU detection, configuration, and optimization.

    Usage:
        manager = GPUManager()
        gpu = manager.auto_detect()
        manager.configure_pytorch()
    """

    def __init__(self):
        self.detected_gpus: List[GPUInfo] = []
        self.primary_gpu: Optional[GPUInfo] = None
        self._pytorch_available = False
        self._check_pytorch()

    def _check_pytorch(self):
        """Check if PyTorch is available."""
        import importlib.util
        self._pytorch_available = importlib.util.find_spec("torch") is not None

    def auto_detect(self) -> Optional[GPUInfo]:
        """
        Auto-detect available GPUs.

        Returns:
            GPUInfo for the best available GPU, or None if no GPU found
        """
        self.detected_gpus = []

        # Try NVIDIA first
        nvidia_gpus = self._detect_nvidia()
        self.detected_gpus.extend(nvidia_gpus)

        # Try AMD ROCm
        amd_gpus = self._detect_amd()
        self.detected_gpus.extend(amd_gpus)

        # Try Apple MPS
        mps_gpu = self._detect_mps()
        if mps_gpu:
            self.detected_gpus.append(mps_gpu)

        # Select primary GPU
        if self.detected_gpus:
            # Prefer NVIDIA, then AMD, then MPS
            for gpu_type in [GPUType.NVIDIA_CUDA, GPUType.AMD_ROCM, GPUType.APPLE_MPS]:
                for gpu in self.detected_gpus:
                    if gpu.gpu_type == gpu_type:
                        self.primary_gpu = gpu
                        return gpu

        # CPU fallback
        self.primary_gpu = GPUInfo(
            gpu_type=GPUType.CPU,
            name="CPU",
            memory_total_mb=0,
            memory_free_mb=0,
            compute_capability="N/A",
            driver_version="N/A"
        )
        return self.primary_gpu

    def _detect_nvidia(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        gpus = []

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpus.append(GPUInfo(
                                gpu_type=GPUType.NVIDIA_CUDA,
                                name=parts[0],
                                memory_total_mb=int(parts[1]),
                                memory_free_mb=int(parts[2]),
                                driver_version=parts[3],
                                compute_capability=parts[4],
                                device_index=i
                            ))

                # Get CUDA version
                cuda_result = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True, text=True, timeout=5
                )
                if cuda_result.returncode == 0:
                    import re
                    match = re.search(r'release (\d+\.\d+)', cuda_result.stdout)
                    if match:
                        for gpu in gpus:
                            gpu.cuda_version = match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return gpus

    def _detect_amd(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi."""
        gpus = []

        try:
            result = subprocess.run(
                ['rocm-smi', '--showid', '--showmeminfo', 'vram', '--csv'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Parse rocm-smi output
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[1:]):  # Skip header
                    if line and not line.startswith('device'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            gpus.append(GPUInfo(
                                gpu_type=GPUType.AMD_ROCM,
                                name=f"AMD GPU {i}",
                                memory_total_mb=int(parts[1]) if parts[1].isdigit() else 0,
                                memory_free_mb=0,  # Would need additional parsing
                                compute_capability="gfx",
                                driver_version="ROCm",
                                device_index=i
                            ))

            # Try to get more details
            info_result = subprocess.run(
                ['rocminfo'],
                capture_output=True, text=True, timeout=10
            )
            if info_result.returncode == 0 and gpus:
                import re
                # Extract GPU name
                match = re.search(r'Name:\s*(\S+)', info_result.stdout)
                if match and gpus:
                    gpus[0].name = match.group(1)

                # Extract gfx version
                gfx_match = re.search(r'gfx\d+', info_result.stdout)
                if gfx_match and gpus:
                    gpus[0].compute_capability = gfx_match.group(0)

        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check ROCm version
        try:
            version_result = subprocess.run(
                ['cat', '/opt/rocm/.info/version'],
                capture_output=True, text=True, timeout=5
            )
            if version_result.returncode == 0:
                for gpu in gpus:
                    gpu.rocm_version = version_result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return gpus

    def _detect_mps(self) -> Optional[GPUInfo]:
        """Detect Apple Metal Performance Shaders (MPS)."""
        if sys.platform != 'darwin':
            return None

        try:
            import torch
            if torch.backends.mps.is_available():
                return GPUInfo(
                    gpu_type=GPUType.APPLE_MPS,
                    name="Apple M-series GPU",
                    memory_total_mb=0,  # MPS shares system memory
                    memory_free_mb=0,
                    compute_capability="MPS",
                    driver_version="Metal"
                )
        except (ImportError, AttributeError):
            pass

        return None

    def configure_pytorch(self) -> str:
        """
        Configure PyTorch to use the detected GPU.

        Returns:
            Device string for PyTorch ('cuda:0', 'mps', 'cpu')
        """
        if not self._pytorch_available:
            return 'cpu'

        import torch

        if self.primary_gpu is None:
            self.auto_detect()

        if self.primary_gpu.gpu_type == GPUType.NVIDIA_CUDA:
            if torch.cuda.is_available():
                device = f'cuda:{self.primary_gpu.device_index}'
                # Configure for optimal performance
                torch.backends.cudnn.benchmark = True
                return device

        elif self.primary_gpu.gpu_type == GPUType.AMD_ROCM:
            if torch.cuda.is_available():  # ROCm uses CUDA API
                device = f'cuda:{self.primary_gpu.device_index}'
                # Set environment for ROCm
                gfx = self.primary_gpu.compute_capability
                if gfx.startswith('gfx'):
                    os.environ['HSA_OVERRIDE_GFX_VERSION'] = gfx[3:].replace('0', '.0.')
                return device

        elif self.primary_gpu.gpu_type == GPUType.APPLE_MPS:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'

        return 'cpu'

    def set_optimal_environment(self):
        """Set optimal environment variables for GPU usage."""
        if self.primary_gpu is None:
            self.auto_detect()

        if self.primary_gpu.gpu_type == GPUType.NVIDIA_CUDA:
            # NVIDIA optimizations
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        elif self.primary_gpu.gpu_type == GPUType.AMD_ROCM:
            # AMD ROCm optimizations
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['GPU_MAX_HW_QUEUES'] = '8'
            os.environ['HIP_VISIBLE_DEVICES'] = str(self.primary_gpu.device_index)

            # Set gfx version if known
            gfx = self.primary_gpu.compute_capability
            if gfx.startswith('gfx'):
                version = gfx[3:]
                # Convert gfx1100 -> 11.0.0
                if len(version) >= 4:
                    os.environ['HSA_OVERRIDE_GFX_VERSION'] = f"{version[0:2]}.{version[2]}.{version[3]}"

    def get_memory_info(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        if not self._pytorch_available:
            return {'total': 0, 'used': 0, 'free': 0}

        import torch

        if torch.cuda.is_available():
            return {
                'total': torch.cuda.get_device_properties(0).total_memory,
                'used': torch.cuda.memory_allocated(),
                'free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            }

        return {'total': 0, 'used': 0, 'free': 0}


def auto_detect_gpu() -> GPUInfo:
    """Convenience function to auto-detect GPU."""
    manager = GPUManager()
    return manager.auto_detect()


def get_gpu_info() -> str:
    """Get formatted GPU information string."""
    manager = GPUManager()
    gpu = manager.auto_detect()

    lines = ["=" * 60, "GPU INFORMATION", "=" * 60]

    if gpu.gpu_type == GPUType.CPU:
        lines.append("No GPU detected - using CPU")
    else:
        lines.append(f"Type:               {gpu.gpu_type.value}")
        lines.append(f"Name:               {gpu.name}")
        lines.append(f"Memory Total:       {gpu.memory_total_mb} MB")
        lines.append(f"Memory Free:        {gpu.memory_free_mb} MB")
        lines.append(f"Compute Capability: {gpu.compute_capability}")
        lines.append(f"Driver Version:     {gpu.driver_version}")

        if gpu.cuda_version:
            lines.append(f"CUDA Version:       {gpu.cuda_version}")
        if gpu.rocm_version:
            lines.append(f"ROCm Version:       {gpu.rocm_version}")

    lines.append("=" * 60)

    # PyTorch status
    try:
        import torch
        lines.append(f"PyTorch Version:    {torch.__version__}")
        lines.append(f"CUDA Available:     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA Device:        {torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("PyTorch:            Not installed")

    lines.append("=" * 60)

    return "\n".join(lines)


def setup_gpu_environment():
    """Set up optimal GPU environment and return device string."""
    manager = GPUManager()
    manager.auto_detect()
    manager.set_optimal_environment()
    return manager.configure_pytorch()
