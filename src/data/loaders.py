"""PyTorch Dataset and DataLoader construction for APTOS screening splits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import (
    E001BaselineTransform,
    E002CLAHENormalizeTransform,
    E004AugmentationTransform,
    get_training_transforms,
    get_validation_transforms,
)

logger = logging.getLogger(__name__)


class APTOSDataset(Dataset):
    """APTOS 2019 PyTorch Dataset with image validation and duplicate quarantine filtering."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        image_dir: Union[str, Path],
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Split CSV manifest not found: {self.csv_path.resolve()}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir.resolve()}")

        raw_df = pd.read_csv(self.csv_path)

        # 1. Check in-file boolean/string column flag
        quarantine_mask = pd.Series(False, index=raw_df.index)
        if "is_conflicting_duplicate" in raw_df.columns:
            quarantine_mask |= (
                (raw_df["is_conflicting_duplicate"] == True)
                | (raw_df["is_conflicting_duplicate"].astype(str).str.strip().str.lower() == "true")
                | (raw_df["is_conflicting_duplicate"] == 1)
            )

        # 2. Check external quarantine manifest if present
        quarantine_csv = self.csv_path.parent / "quarantined_conflicting_duplicates.csv"
        if quarantine_csv.is_file():
            q_df = pd.read_csv(quarantine_csv)
            id_col = "id_code" if "id_code" in q_df.columns else q_df.columns[0]
            quarantine_mask |= raw_df["id_code"].isin(q_df[id_col])

        filtered_df = raw_df[~quarantine_mask].copy()
        quarantined_count = len(raw_df) - len(filtered_df)
        if quarantined_count > 0:
            logger.info(
                "Quarantined %d conflicting duplicates from %s (Active cohort: %d)",
                quarantined_count,
                self.csv_path.name,
                len(filtered_df),
            )
        self.df = filtered_df.reset_index(drop=True)

        for col in ["id_code", "diagnosis"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing mandatory column '{col}' in {self.csv_path}")

        try:
            self.df["diagnosis"] = self.df["diagnosis"].astype(int)
        except Exception as e:
            raise ValueError(f"Diagnosis column contains non-integer values: {e}")

        invalid_labels = self.df[~self.df["diagnosis"].isin([0, 1, 2, 3, 4])]
        if not invalid_labels.empty:
            raise ValueError(f"Found invalid ICDR diagnoses outside [0, 4]: {invalid_labels}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        label = int(row["diagnosis"])

        image_path = self.image_dir / f"{id_code}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Expected fundus image not found on disk: {image_path.resolve()}")

        try:
            with Image.open(image_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
        except Exception as exc:
            raise IOError(f"Corrupted or unreadable image file {image_path}: {exc}") from exc

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            arr = np.array(img_rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)

        target = torch.tensor(label, dtype=torch.long)
        return tensor, target, id_code


def create_aptos_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct Train, Validation, and Test DataLoaders for APTOS 2019."""
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"
    test_csv = data_dir / "processed" / "aptos" / "test.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())

    train_transform = get_training_transforms(base_config)
    eval_transform = get_validation_transforms(base_config)

    train_dataset = APTOSDataset(train_csv, raw_images, transform=train_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=eval_transform)
    test_dataset = APTOSDataset(test_csv, raw_images, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_e002_clahe_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct Train and Validation DataLoaders with E002 CLAHE transforms.
    CRITICAL GOVERNANCE: test.csv is explicitly NOT loaded to protect the held-out benchmark.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())

    clahe_cfg = exp_config.get("preprocessing_variants", {}).get("clahe", {})
    clip_limit = clahe_cfg.get("clip_limit", 2.0)
    tile_grid_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
    image_size = tuple(base_config.get("data", {}).get("image_size", [512, 512]))

    clahe_transform = E002CLAHENormalizeTransform(
        target_size=image_size,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    train_dataset = APTOSDataset(train_csv, raw_images, transform=clahe_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=clahe_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_e003_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct Train and Validation DataLoaders for experiment E003 (Class Balance).
    Uses E001 baseline transforms (crop -> resize 512x512 -> [0, 1]).
    CRITICAL GOVERNANCE: test.csv is strictly NOT loaded to protect the held-out benchmark.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())

    train_transform = get_training_transforms(base_config)
    val_transform = get_validation_transforms(base_config)

    train_dataset = APTOSDataset(train_csv, raw_images, transform=train_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_e004_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct Train and Validation DataLoaders for experiment E004 (Augmentation).
    Train DataLoader uses E004AugmentationTransform.
    Validation DataLoader uses deterministic E001BaselineTransform.
    CRITICAL GOVERNANCE: test.csv is strictly NOT loaded to protect the held-out benchmark.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())
    image_size = tuple(base_config.get("data", {}).get("image_size", [512, 512]))

    aug_cfg = exp_config.get("augmentation", {})
    train_transform = E004AugmentationTransform(
        target_size=image_size,
        hflip_p=aug_cfg.get("horizontal_flip", {}).get("probability", 0.5),
        rotation_p=aug_cfg.get("rotation", {}).get("probability", 0.7),
        max_rotation_deg=aug_cfg.get("rotation", {}).get("max_degrees", 10.0),
        affine_p=aug_cfg.get("affine", {}).get("probability", 0.5),
        scale_range=tuple(aug_cfg.get("affine", {}).get("scale_range", [0.96, 1.04])),
        translate_pct=aug_cfg.get("affine", {}).get("translate_pct", 0.02),
        jitter_p=aug_cfg.get("color_jitter", {}).get("probability", 0.5),
        brightness_range=tuple(aug_cfg.get("color_jitter", {}).get("brightness_factor_range", [0.95, 1.05])),
        contrast_range=tuple(aug_cfg.get("color_jitter", {}).get("contrast_factor_range", [0.95, 1.05])),
    )

    val_transform = E001BaselineTransform(target_size=image_size)

    train_dataset = APTOSDataset(train_csv, raw_images, transform=train_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_e005_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct Train and Validation DataLoaders for experiment E005 (Architecture Comparison).
    Train DataLoader uses E004AugmentationTransform.
    Validation DataLoader uses deterministic E001BaselineTransform.
    CRITICAL GOVERNANCE: test.csv is strictly NOT loaded to protect the held-out benchmark.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())
    image_size = tuple(base_config.get("data", {}).get("image_size", [512, 512]))

    # Identical active augmentation pipeline from E004
    train_transform = E004AugmentationTransform(
        target_size=image_size,
        hflip_p=0.5,
        rotation_p=0.7,
        max_rotation_deg=10.0,
        affine_p=0.5,
        scale_range=(0.96, 1.04),
        translate_pct=0.02,
        jitter_p=0.5,
        brightness_range=(0.95, 1.05),
        contrast_range=(0.95, 1.05),
    )
    val_transform = E001BaselineTransform(target_size=image_size)

    train_dataset = APTOSDataset(train_csv, raw_images, transform=train_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader

def create_e006_data_loaders(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct Train and Validation DataLoaders for E006 Ordinal Regression.
    Uses E004 conservative augmentation for training and deterministic baseline for validation.
    Enforces active validation cohort == 572 (29 quarantined conflicting duplicates excluded).
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir = root / "data"
    raw_images = data_dir / "raw" / "aptos" / "train_images"

    train_csv = data_dir / "processed" / "aptos" / "train.csv"
    val_csv = data_dir / "processed" / "aptos" / "validation.csv"

    batch_size = exp_config.get("training", {}).get("batch_size", 16)
    device_name = base_config.get("compute", {}).get("device", "cpu")
    num_workers = base_config.get("compute", {}).get("num_workers", 2)
    if device_name == "mps" or torch.backends.mps.is_available():
        num_workers = min(num_workers, 2)

    pin_memory = (device_name == "cuda" and torch.cuda.is_available())
    image_size = tuple(base_config.get("data", {}).get("image_size", [512, 512]))

    train_transform = E004AugmentationTransform(target_size=image_size)
    val_transform = E001BaselineTransform(target_size=image_size)

    train_dataset = APTOSDataset(train_csv, raw_images, transform=train_transform)
    val_dataset = APTOSDataset(val_csv, raw_images, transform=val_transform)

    # FAIL-FAST COHORT VALIDATION GUARD
    EXPECTED_ACTIVE_VAL_COUNT = 601
    if len(val_dataset) != EXPECTED_ACTIVE_VAL_COUNT:
        raise RuntimeError(
            f"VALIDATION INTEGRITY VIOLATION: Expected exactly {EXPECTED_ACTIVE_VAL_COUNT} "
            f"active validation samples, but loaded {len(val_dataset)}. Halting."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader

