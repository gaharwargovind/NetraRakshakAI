"""Unit tests for random seed initialization and system environment profiling."""

import random
import numpy as np
import torch
from src.utils.seed import seed_everything
from src.utils.system_info import collect_system_fingerprint


def test_seed_everything_deterministic():
    """Verify that setting seeds enforces reproducible pseudo-random sequences."""
    seed_val = 1234
    seed_everything(seed=seed_val, deterministic=True)
    
    py_rand1 = [random.random() for _ in range(5)]
    np_rand1 = np.random.rand(5).tolist()
    th_rand1 = torch.rand(5).tolist()
    
    # Re-seed with identical value
    seed_everything(seed=seed_val, deterministic=True)
    
    py_rand2 = [random.random() for _ in range(5)]
    np_rand2 = np.random.rand(5).tolist()
    th_rand2 = torch.rand(5).tolist()
    
    assert py_rand1 == py_rand2, "Python random generator failed reproducibility test"
    assert np_rand1 == np_rand2, "NumPy random generator failed reproducibility test"
    assert th_rand1 == th_rand2, "PyTorch random generator failed reproducibility test"


def test_system_fingerprint_collection():
    """Verify that system profiling returns mandatory environmental fields."""
    fingerprint = collect_system_fingerprint()
    assert isinstance(fingerprint, dict)
    for field in ["os", "python_version", "pytorch_version", "cuda_available", "device_name"]:
        assert field in fingerprint, f"Missing system fingerprint field: {field}"