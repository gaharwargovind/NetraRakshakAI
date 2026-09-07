import sys, json, hashlib
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

e007_ckpt = ROOT / "models/checkpoints/E007_best_model.pt"
assert e007_ckpt.is_file(), "E007 checkpoint missing"
e007_sha = sha256_file(e007_ckpt)

train_df = pd.read_csv(ROOT / "data/processed/aptos/train.csv")
val_df = pd.read_csv(ROOT / "data/processed/aptos/validation.csv")
assert len(val_df) == 601, f"Expected 601 validation records, found {len(val_df)}"

counts = train_df["diagnosis"].value_counts().sort_index().to_dict()
total = len(train_df)
alpha = [round(float(total / (5.0 * counts[c])), 4) for c in range(5)]

exp_dir = ROOT / "experiments/E009_class_balanced_focal"
exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / ".e007_baseline_sha").write_text(e007_sha)
(exp_dir / ".val_baseline_sha").write_text(sha256_file(ROOT / "data/processed/aptos/validation.csv"))
(exp_dir / ".test_baseline_sha").write_text(sha256_file(ROOT / "data/processed/aptos/test.csv"))

print("=" * 65)
print("       EXPERIMENT E009 PRE-FLIGHT AUDIT & ALPHA VALUES")
print("=" * 65)
print(f"E007 Checkpoint SHA-256: {e007_sha}")
print(f"Validation records:      {len(val_df)} (strictly isolated)")
print(f"Training records:        {total}")
print(f"Training counts:         {counts}")
print(f"Alpha formula:           alpha_c = N_train / (5 * N_c)")
print(f"Derived Alpha weights:   {alpha}")
for c in range(5):
    print(f"  - Grade {c}: {alpha[c]:.4f} (Support: {counts[c]})")
print("=" * 65)
