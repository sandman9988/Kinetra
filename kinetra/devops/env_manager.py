"""
Environment Manager
===================

Permanently resolve Python path and environment issues:
- Virtual environment management
- Python path configuration
- Dependency verification
- System path setup
"""

import os
import shutil
import subprocess
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvInfo:
    """Environment information."""
    python_path: str
    python_version: str
    pip_path: str
    virtual_env: Optional[str]
    site_packages: List[str]
    path_dirs: List[str]
    pythonpath: List[str]
    installed_packages: Dict[str, str]


class EnvManager:
    """
    Manage Python environment configuration.

    Usage:
        manager = EnvManager()
        manager.setup_environment()
        manager.verify_environment()
    """

    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()
        self.venv_path = self.workspace / ".venv"
        self._original_path = os.environ.get('PATH', '')
        self._original_pythonpath = os.environ.get('PYTHONPATH', '')

    def get_env_info(self) -> EnvInfo:
        """Get current environment information."""
        # Get Python info
        python_path = sys.executable
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get pip path
        pip_path = shutil.which('pip3') or shutil.which('pip') or ''

        # Get virtual env
        virtual_env = os.environ.get('VIRTUAL_ENV')

        # Get site packages
        site_packages = [str(p) for p in sys.path if 'site-packages' in str(p)]

        # Get PATH directories
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)

        # Get PYTHONPATH
        pythonpath = os.environ.get('PYTHONPATH', '').split(os.pathsep) if os.environ.get('PYTHONPATH') else []

        # Get installed packages
        installed = self._get_installed_packages()

        return EnvInfo(
            python_path=python_path,
            python_version=python_version,
            pip_path=pip_path,
            virtual_env=virtual_env,
            site_packages=site_packages,
            path_dirs=path_dirs,
            pythonpath=pythonpath,
            installed_packages=installed
        )

    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed packages and versions."""
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    packages[pkg['name'].lower()] = pkg['version']
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        return packages

    def create_venv(self, force: bool = False) -> Tuple[bool, str]:
        """Create a virtual environment."""
        if self.venv_path.exists() and not force:
            return True, f"Virtual environment already exists at {self.venv_path}"

        try:
            if force and self.venv_path.exists():
                shutil.rmtree(self.venv_path)

            subprocess.run(
                [sys.executable, '-m', 'venv', str(self.venv_path)],
                check=True, capture_output=True, timeout=120
            )

            # Install pip in venv
            venv_python = self.venv_path / 'bin' / 'python'
            subprocess.run(
                [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'],
                check=True, capture_output=True, timeout=120
            )

            return True, f"Created virtual environment at {self.venv_path}"
        except subprocess.SubprocessError as e:
            return False, f"Failed to create venv: {e}"

    def activate_venv(self) -> Tuple[bool, str]:
        """Activate the virtual environment (for current process)."""
        if not self.venv_path.exists():
            return False, "Virtual environment does not exist"

        venv_bin = self.venv_path / 'bin'
        venv_python = venv_bin / 'python'

        if not venv_python.exists():
            return False, "Virtual environment is corrupted"

        # Update PATH
        path = os.environ.get('PATH', '')
        if str(venv_bin) not in path:
            os.environ['PATH'] = f"{venv_bin}{os.pathsep}{path}"

        # Update VIRTUAL_ENV
        os.environ['VIRTUAL_ENV'] = str(self.venv_path)

        return True, f"Activated virtual environment: {self.venv_path}"

    def setup_environment(self, create_venv: bool = True) -> Tuple[bool, str]:
        """
        Set up complete Python environment.

        Steps:
        1. Create/activate virtual environment
        2. Add workspace to PYTHONPATH
        3. Set up PATH correctly
        4. Create environment setup scripts
        """
        messages = []

        # Add workspace to PYTHONPATH
        pythonpath = os.environ.get('PYTHONPATH', '')
        workspace_str = str(self.workspace)
        if workspace_str not in pythonpath:
            os.environ['PYTHONPATH'] = f"{workspace_str}{os.pathsep}{pythonpath}" if pythonpath else workspace_str
            messages.append(f"Added {workspace_str} to PYTHONPATH")

        # Ensure workspace is in sys.path
        if workspace_str not in sys.path:
            sys.path.insert(0, workspace_str)
            messages.append("Added workspace to sys.path")

        # Create virtual environment if requested
        if create_venv:
            if not self.venv_path.exists():
                success, msg = self.create_venv()
                if not success:
                    return False, msg
                messages.append(msg)

            success, msg = self.activate_venv()
            if success:
                messages.append(msg)

        # Generate permanent setup scripts
        self._generate_setup_scripts()
        messages.append("Generated environment setup scripts")

        return True, "; ".join(messages)

    def _generate_setup_scripts(self):
        """Generate setup scripts for persistent configuration."""
        # Bash script
        bash_script = self.workspace / "env_setup.sh"
        bash_content = f'''#!/bin/bash
# Kinetra Environment Setup
# Source this file: source env_setup.sh

# Set workspace
export KINETRA_WORKSPACE="{self.workspace}"

# Add to PYTHONPATH
export PYTHONPATH="${{KINETRA_WORKSPACE}}:${{PYTHONPATH}}"

# Activate virtual environment if it exists
if [ -d "{self.venv_path}" ]; then
    source "{self.venv_path}/bin/activate"
fi

# Add local bin to PATH
export PATH="${{KINETRA_WORKSPACE}}/scripts:$PATH"

# GPU environment variables (AMD ROCm)
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
export GPU_MAX_HW_QUEUES="8"

# Python optimizations
export PYTHONOPTIMIZE="1"
export PYTHONDONTWRITEBYTECODE="1"

echo "Kinetra environment loaded"
echo "Python: $(which python3)"
echo "Workspace: $KINETRA_WORKSPACE"
'''

        with open(bash_script, 'w') as f:
            f.write(bash_content)
        os.chmod(bash_script, 0o755)

        # Python startup file
        python_startup = self.workspace / ".pythonstartup"
        startup_content = f'''# Kinetra Python Startup
import sys
import os

# Add workspace to path
workspace = "{self.workspace}"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

# Set PYTHONPATH
pythonpath = os.environ.get("PYTHONPATH", "")
if workspace not in pythonpath:
    os.environ["PYTHONPATH"] = f"{{workspace}}:{{pythonpath}}" if pythonpath else workspace

print(f"[Kinetra] Workspace: {{workspace}}")
'''

        with open(python_startup, 'w') as f:
            f.write(startup_content)

        # Create .envrc for direnv users
        envrc = self.workspace / ".envrc"
        envrc_content = f'''# Kinetra direnv configuration
# Install direnv: https://direnv.net/

export KINETRA_WORKSPACE="{self.workspace}"
export PYTHONPATH="${{KINETRA_WORKSPACE}}:${{PYTHONPATH}}"
export PATH="${{KINETRA_WORKSPACE}}/scripts:$PATH"

# Activate venv if available
if [ -f "{self.venv_path}/bin/activate" ]; then
    source "{self.venv_path}/bin/activate"
fi
'''

        with open(envrc, 'w') as f:
            f.write(envrc_content)

    def install_requirements(self, requirements_file: str = "requirements.txt") -> Tuple[bool, str]:
        """Install requirements from file."""
        req_path = self.workspace / requirements_file

        if not req_path.exists():
            return False, f"Requirements file not found: {req_path}"

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', str(req_path)],
                capture_output=True, text=True, timeout=600
            )

            if result.returncode == 0:
                return True, "Requirements installed successfully"
            else:
                return False, f"Installation failed: {result.stderr}"
        except subprocess.SubprocessError as e:
            return False, f"Installation error: {e}"

    def verify_environment(self) -> Dict[str, any]:
        """Verify environment is correctly configured."""
        info = self.get_env_info()
        issues = []

        # Check Python version
        major, minor = map(int, info.python_version.split('.')[:2])
        if major < 3 or (major == 3 and minor < 8):
            issues.append(f"Python version {info.python_version} is too old (need 3.8+)")

        # Check workspace in PYTHONPATH
        if str(self.workspace) not in info.pythonpath:
            issues.append("Workspace not in PYTHONPATH")

        # Check critical packages
        critical_packages = ['numpy', 'pandas', 'torch', 'scikit-learn']
        missing = [p for p in critical_packages if p not in info.installed_packages]
        if missing:
            issues.append(f"Missing packages: {', '.join(missing)}")

        return {
            "valid": len(issues) == 0,
            "python_version": info.python_version,
            "python_path": info.python_path,
            "virtual_env": info.virtual_env,
            "packages_installed": len(info.installed_packages),
            "issues": issues
        }

    def get_report(self) -> str:
        """Generate environment report."""
        info = self.get_env_info()
        verification = self.verify_environment()

        lines = ["=" * 60, "ENVIRONMENT REPORT", "=" * 60]
        lines.append(f"Python:        {info.python_path}")
        lines.append(f"Version:       {info.python_version}")
        lines.append(f"Pip:           {info.pip_path}")
        lines.append(f"Virtual Env:   {info.virtual_env or 'None'}")
        lines.append(f"Packages:      {len(info.installed_packages)} installed")
        lines.append("")

        lines.append("PYTHONPATH:")
        for p in info.pythonpath:
            lines.append(f"  - {p}")

        lines.append("")
        lines.append("Site Packages:")
        for p in info.site_packages[:3]:
            lines.append(f"  - {p}")

        lines.append("")
        lines.append("-" * 60)

        if verification['valid']:
            lines.append("✅ Environment is correctly configured")
        else:
            lines.append("❌ Environment issues detected:")
            for issue in verification['issues']:
                lines.append(f"  - {issue}")

        lines.append("=" * 60)

        return "\n".join(lines)


def setup_environment() -> Tuple[bool, str]:
    """Convenience function to set up environment."""
    manager = EnvManager()
    return manager.setup_environment()


def verify_environment() -> Dict[str, any]:
    """Convenience function to verify environment."""
    manager = EnvManager()
    return manager.verify_environment()


def get_env_report() -> str:
    """Get environment report."""
    manager = EnvManager()
    return manager.get_report()


def ensure_workspace_importable():
    """Ensure the workspace directory is importable."""
    workspace = Path.cwd()
    workspace_str = str(workspace)

    # Add to sys.path
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)

    # Add to PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    if workspace_str not in pythonpath:
        os.environ['PYTHONPATH'] = f"{workspace_str}{os.pathsep}{pythonpath}" if pythonpath else workspace_str

    return workspace_str
