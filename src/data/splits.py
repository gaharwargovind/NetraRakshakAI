"""
Leakage-safe dataset splitting for diabetic retinopathy datasets.

For APTOS:
- No patient identifier is available.
- Exact image-content groups are therefore used as the leakage-prevention unit.
- Conflicting exact-duplicate groups are excluded from development splits.
- Consistent duplicate copies are kept in the same split.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading the complete image into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_exact_duplicate_groups(
    image_dir: Path,
    id_col: str = "id_code",
) -> pd.DataFrame:
    """
    Build exact-content groups using file size followed by SHA-256.

    Files with unique byte size cannot be exact duplicates, so SHA-256
    is calculated only for files sharing a byte size.

    Returns a dataframe with:
        id_code
        sha256
        duplicate_group
    """
    image_dir = Path(image_dir)

    files = sorted(image_dir.glob("*.png"))

    if not files:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    size_buckets = {}

    for path in files:
        size = path.stat().st_size
        size_buckets.setdefault(size, []).append(path)

    records = []

    for size, candidates in size_buckets.items():
        if len(candidates) == 1:
            path = candidates[0]
            digest = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            records.append(
                {
                    id_col: path.stem,
                    "sha256": digest,
                    "file_size": size,
                }
            )
            continue

        for path in candidates:
            digest = _sha256_file(path)

            records.append(
                {
                    id_col: path.stem,
                    "sha256": digest,
                    "file_size": size,
                }
            )

    groups = pd.DataFrame(records)

    groups["duplicate_group"] = groups["sha256"].factorize(
        sort=True
    )[0]

    return groups


def create_patient_stratified_split(
    df: pd.DataFrame,
    patient_col: Optional[str],
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create a stratified group-aware train/validation/test split.

    If patient_col is supplied and exists, it is used as the grouping unit.

    Otherwise, `id_code` must be present and is used as the grouping unit.
    This fallback is intended for datasets where patient IDs are unavailable.

    Note:
        For APTOS, exact duplicate grouping should be performed before calling
        this function and the resulting group identifier should be supplied
        through the `group_col` argument in the wrapper below.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    if patient_col is not None:
        if patient_col not in df.columns:
            raise ValueError(f"Missing grouping column: {patient_col}")
        group_col = patient_col
    else:
        if "id_code" not in df.columns:
            raise ValueError(
                "No patient column supplied and 'id_code' is missing."
            )
        group_col = "id_code"

    working = df.reset_index(drop=True).copy()

    groups = working[group_col]
    labels = working[label_col]

    # First split: train vs temporary (validation + test)
    train_fraction = train_ratio
    temp_fraction = val_ratio + test_ratio

    first_split = StratifiedGroupKFold(
        n_splits=round(1.0 / temp_fraction),
        shuffle=True,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        first_split.split(
            working,
            y=labels,
            groups=groups,
        )
    )

    train_df = working.iloc[train_idx].copy()
    temp_df = working.iloc[temp_idx].copy()

    # Second split: validation vs test.
    # They are approximately equal because val_ratio == test_ratio
    # in the project's default configuration.
    relative_val = val_ratio / temp_fraction

    if abs(relative_val - 0.5) > 1e-6:
        raise ValueError(
            "Current group split implementation expects equal "
            "validation and test ratios."
        )

    second_split = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed + 1,
    )

    val_idx, test_idx = next(
        second_split.split(
            temp_df,
            y=temp_df[label_col],
            groups=temp_df[group_col],
        )
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def create_aptos_split(
    csv_path: Path,
    image_dir: Path,
    conflict_manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create the official APTOS development split.

    Conflicting exact-duplicate groups are excluded.
    Exact duplicate groups are kept intact across splits.
    """

    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    conflict_manifest_path = Path(conflict_manifest_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(csv_path)

    required_columns = {"id_code", "diagnosis"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_groups = build_exact_duplicate_groups(image_dir)

    merged = df.merge(
        duplicate_groups,
        on="id_code",
        how="left",
        validate="one_to_one",
    )

    if merged["sha256"].isna().any():
        missing_images = merged.loc[
            merged["sha256"].isna(), "id_code"
        ].tolist()

        raise ValueError(
            f"Missing image hashes for {len(missing_images)} records."
        )

    conflicts = pd.read_csv(conflict_manifest_path)

    conflict_ids = set(conflicts["id_code"].astype(str))

    merged["is_conflicting_duplicate"] = (
        merged["id_code"].astype(str).isin(conflict_ids)
    )

    eligible = merged.loc[
        ~merged["is_conflicting_duplicate"]
    ].copy()

    quarantined = merged.loc[
        merged["is_conflicting_duplicate"]
    ].copy()

    logger.info("Total records: %d", len(merged))
    logger.info("Quarantined records: %d", len(quarantined))
    logger.info("Eligible records: %d", len(eligible))

    # IMPORTANT:
    # The SHA-256 hash is the grouping unit for exact-content leakage
    # prevention.
    splits = create_patient_stratified_split(
        eligible,
        patient_col="sha256",
        label_col="diagnosis",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_df.to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )

    quarantined.to_csv(
        output_dir / "quarantined_conflicting_duplicates.csv",
        index=False,
    )

    duplicate_groups.to_csv(
        output_dir / "exact_duplicate_groups.csv",
        index=False,
    )

    return splits
