"""Configuration management and schema validation utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Union
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely parse and load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary representation of the configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML syntax parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"Configuration root must be a mapping, got {type(cfg).__name__}")
            logger.debug("Configuration successfully loaded from %s", path.name)
            return cfg
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML file {path}: {exc}") from exc


def validate_config_structure(cfg: Dict[str, Any], required_keys: list[str]) -> bool:
    """
    Validate that all mandatory top-level keys exist within the configuration mapping.
    
    Args:
        cfg: Configuration dictionary.
        required_keys: List of expected top-level key names.
        
    Returns:
        True if all required keys are present.
        
    Raises:
        KeyError: If any mandatory key is missing.
    """
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(f"Configuration is missing mandatory keys: {missing}")
    return True