"""Contract and behavioral tests for data partitioning and leakage prevention."""

import pandas as pd
from src.data.splits import create_patient_stratified_split


def test_dataset_split_interface():
    """
    Verify create_patient_stratified_split partitions data without patient leakage,
    retains all records, and outputs the expected partition dictionary.
    """
    records = []
    for p_idx in range(20):
        patient_id = f"P{p_idx:03d}"
        diagnosis = p_idx % 5  # Covers ICDR grades 0..4
        # Multi-capture representation: 2 images per patient
        records.append({"id_code": f"{patient_id}_OD", "patient_id": patient_id, "diagnosis": diagnosis})
        records.append({"id_code": f"{patient_id}_OS", "patient_id": patient_id, "diagnosis": diagnosis})

    dummy_df = pd.DataFrame(records)

    splits = create_patient_stratified_split(
        df=dummy_df,
        patient_col="patient_id",
        label_col="diagnosis",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # 1. Output schema verification
    assert isinstance(splits, dict), "Split output must be a dictionary of DataFrames."
    val_key = "val" if "val" in splits else "validation"
    assert "train" in splits, "Split output missing 'train' partition."
    assert val_key in splits, "Split output missing validation partition."
    assert "test" in splits, "Split output missing 'test' partition."

    train_df = splits["train"]
    val_df = splits[val_key]
    test_df = splits["test"]

    # 2. Record conservation
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == len(dummy_df), f"Expected {len(dummy_df)} total records, found {total_split_rows}."

    # 3. Patient-level isolation (zero leakage across partitions)
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert train_patients.isdisjoint(val_patients), "Patient overlap detected between train and validation."
    assert train_patients.isdisjoint(test_patients), "Patient overlap detected between train and test."
    assert val_patients.isdisjoint(test_patients), "Patient overlap detected between validation and test."