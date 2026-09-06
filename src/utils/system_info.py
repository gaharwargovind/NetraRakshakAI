"""System environmental provenance and hardware diagnostic profiling."""

import logging
import platform
import sys
from typing import Any, Dict
import torch

logger = logging.getLogger(__name__)


def collect_system_fingerprint() -> Dict[str, Any]:
    """
    Collect comprehensive environment metadata for experiment auditability.
    
    Returns:
        Dictionary containing OS, Python, PyTorch, and GPU hardware states.
    """
    fingerprint: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
    }
    logger.info("System environment profiled: %s on %s", fingerprint["python_version"], fingerprint["device_name"])
    return fingerprint