"""Reproducibility utilities for deterministic pseudo-random number generation."""

import os
import random
import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set random seeds across Python, NumPy, and PyTorch for full reproducibility.
    
    Args:
        seed: Integer seed value.
        deterministic: If True, configures CuDNN for deterministic operation.
        
    Returns:
        The active seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic backend mode enforced with seed: %d", seed)
    else:
        logger.info("Random seed initialized to %d (non-deterministic mode)", seed)

    return seed