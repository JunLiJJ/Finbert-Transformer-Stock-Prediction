# tests/test_data_validation.py
import os
import glob
import pandas as pd
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)


def _csv_paths():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))


# -------- 1) 基础：CSV 可读 & 非空 --------
@pytest.mark.parametrize("csv_path", _csv_paths() or ["__NO_CSV__"])
def test_csv_readable_and_nonempty(csv_path):
    if csv_path == "__NO_CSV__":
        pytest.skip("No CSV files found in data/; skipping CSV tests.")
    df = pd.read_csv(csv_path)
    assert not df.empty, f"{os.path.basename(csv_path)} should not be empty"
    assert df.columns.size > 0, f"{os.path.basename(csv_path)} has no columns"


# -------- 2) 行列对齐检查 --------
@pytest.mark.parametrize("csv_path", _csv_paths() or ["__NO_CSV__"])
def test_no_empty_rows_and_unique_columns(csv_path):
    if csv_path == "__NO_CSV__":
        pytest.skip("No CSV files found in data/; skipping row/col tests.")
    df = pd.read_csv(csv_path)

    # 没有完全空的行
    empty_rows = df.isna().all(axis=1).sum()
    assert empty_rows == 0, f"{os.path.basename(csv_path)} has {empty_rows} completely empty rows"

    # 列名唯一
    duplicates = df.columns[df.columns.duplicated()].tolist()
    assert not duplicates, f"{os.path.basename(csv_path)} has duplicate columns: {duplicates}"


# -------- 3) 文件体积/行数下限 --------
@pytest.mark.parametrize("csv_path", _csv_paths() or ["__NO_CSV__"])
def test_minimum_rows(csv_path):
    if csv_path == "__NO_CSV__":
        pytest.skip("No CSV files found in data/; skipping row-count test.")
    df = pd.read_csv(csv_path)
    assert len(df) >= 10, f"{os.path.basename(csv_path)} has too few rows (<10); adjust threshold if intended."


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
