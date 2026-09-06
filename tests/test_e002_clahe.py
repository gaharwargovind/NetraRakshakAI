def test_e002_data_loaders_isolation():
    """Verify E002 loader loads Train and Validation only, strictly excluding test.csv."""
    from pathlib import Path
    import pandas as pd
    import torch
    from src.data.loaders import create_e002_clahe_data_loaders
    from src.utils.config import load_config

    base_cfg = load_config("configs/base.yaml")
    exp_cfg = load_config("configs/experiments/preprocessing.yaml")

    # Force num_workers=0 for tests on macOS to prevent zombie worker threads
    base_cfg["compute"]["num_workers"] = 0

    train_loader, val_loader = create_e002_clahe_data_loaders(base_cfg, exp_cfg)

    # Ingest manifests and compute active non-quarantined sample counts
    train_csv = Path("data/processed/aptos/train.csv")
    val_csv = Path("data/processed/aptos/validation.csv")
    test_csv = Path("data/processed/aptos/test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    expected_train_count = len(train_df[train_df.get("is_conflicting_duplicate", False) != True])
    expected_val_count = len(val_df[val_df.get("is_conflicting_duplicate", False) != True])

    # Verify active dataset lengths
    assert len(train_loader.dataset) == expected_train_count
    assert len(val_loader.dataset) == expected_val_count

    # Verify test set isolation (zero leakage into train or validation loaders)
    train_ids = set(train_loader.dataset.df["id_code"])
    val_ids = set(val_loader.dataset.df["id_code"])
    test_ids = set(test_df["id_code"])

    assert train_ids.isdisjoint(test_ids), "Test set IDs leaked into training loader"
    assert val_ids.isdisjoint(test_ids), "Test set IDs leaked into validation loader"

    # Verify batch tensor properties
    first_batch, labels, ids = next(iter(train_loader))
    assert first_batch.shape == (16, 3, 512, 512)
    assert len(labels) == 16
    assert torch.all((labels >= 0) & (labels <= 4))