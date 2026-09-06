# FILE: tools/audit_validation_partition.py
"""
Diagnostic Script: Validation Partition Integrity & Exclusion Audit.
SIH Problem Statement: SIH26038 — Explainable AI for DR Screening in Rural India.

Non-destructive audit utility that:
1. Ingests train.csv, validation.csv, and test.csv.
2. Traces record filtering inside APTOSDataset.
3. Identifies the exact records excluded and the exclusion mechanism.
4. Verifies physical image existence and decodability on disk.
5. Emits a deterministic integrity report without modifying any files.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_split_file(
    split_csv_path: Path,
    image_dir: Path,
    split_name: str,
) -> Dict[str, Any]:
    """Audit a single split manifest for raw vs active records and image readability."""
    if not split_csv_path.is_file():
        raise FileNotFoundError(f"{split_name} manifest not found: {split_csv_path}")

    raw_df = pd.read_csv(split_csv_path)
    total_records = len(raw_df)

    has_duplicate_col = "is_conflicting_duplicate" in raw_df.columns
    if has_duplicate_col:
        # Match APTOSDataset filtering logic exactly:
        # filtered_df = raw_df[raw_df["is_conflicting_duplicate"] != True]
        quarantined_mask = (raw_df["is_conflicting_duplicate"] == True) | (raw_df["is_conflicting_duplicate"] == "True")
        excluded_df = raw_df[quarantined_mask].copy()
        included_df = raw_df[~quarantined_mask].copy()
    else:
        excluded_df = pd.DataFrame()
        included_df = raw_df.copy()

    # Verify physical file existence and decodability
    missing_files = []
    unreadable_files = []

    for _, row in raw_df.iterrows():
        id_code = str(row["id_code"])
        img_path = image_dir / f"{id_code}.png"
        if not img_path.is_file():
            missing_files.append(id_code)
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            unreadable_files.append(id_code)

    return {
        "split_name": split_name,
        "total_manifest_records": total_records,
        "active_evaluated_records": len(included_df),
        "excluded_records_count": len(excluded_df),
        "has_quarantine_column": has_duplicate_col,
        "excluded_records": excluded_df.to_dict(orient="records"),
        "physical_missing_count": len(missing_files),
        "unreadable_count": len(unreadable_files),
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
        "included_class_distribution": (
            included_df["diagnosis"].value_counts().sort_index().to_dict()
            if "diagnosis" in included_df.columns else {}
        ),
        "excluded_class_distribution": (
            excluded_df["diagnosis"].value_counts().sort_index().to_dict()
            if "diagnosis" in excluded_df.columns else {}
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit split files and dataset exclusion logic")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output-json", type=str, default="data/dataset_audit/validation_exclusion_audit.json", help="Output path for audit JSON")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "data"
    proc_dir = data_dir / "processed" / "aptos"
    image_dir = data_dir / "raw" / "aptos" / "train_images"

    logger.info("Auditing dataset split partitions against raw images...")

    results = {}
    for split_name in ["train", "validation", "test"]:
        csv_path = proc_dir / f"{split_name}.csv"
        results[split_name] = audit_split_file(csv_path, image_dir, split_name)

    val_res = results["validation"]
    logger.info("--- VALIDATION PARTITION SUMMARY ---")
    logger.info("Total rows in validation.csv: %d", val_res["total_manifest_records"])
    logger.info("Active evaluated samples:     %d", val_res["active_evaluated_records"])
    logger.info("Excluded records (quarantine): %d", val_res["excluded_records_count"])
    logger.info("Physical missing images:       %d", val_res["physical_missing_count"])
    logger.info("Corrupted/unreadable images:   %d", val_res["unreadable_count"])

    out_path = root / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Full non-destructive audit written to %s", out_path)


if __name__ == "__main__":
    main()