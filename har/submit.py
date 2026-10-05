from __future__ import annotations

from pathlib import Path

import pandas as pd

from har.constants import NUM_CLASSES, NUM_TEST_CLIPS, SUBMISSION_COLUMNS, TEST_ID_FMT, TEST_PATH_PREFIX


def submission_path_for_id(idx: int) -> str:
    return f"{TEST_PATH_PREFIX}/{TEST_ID_FMT.format(idx=idx)}/"


def default_rows() -> pd.DataFrame:
    paths = [submission_path_for_id(i) for i in range(1, NUM_TEST_CLIPS + 1)]
    return pd.DataFrame({"path": paths, "prediction": [0] * NUM_TEST_CLIPS})


def read_test_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    path_col = cols.get("path")
    if path_col is None:
        id_col = cols.get("id")
        if id_col is None:
            raise ValueError(f"{csv_path} has no path/id column: {list(df.columns)}")
        paths = []
        for raw in df[id_col].astype(str):
            name = raw.strip().replace("\\", "/").rstrip("/")
            if "/" not in name:
                name = f"{TEST_PATH_PREFIX}/{name}"
            paths.append(name + "/")
        pred_col = cols.get("prediction") or cols.get("label")
        preds = df[pred_col] if pred_col else [pd.NA] * len(df)
        return pd.DataFrame({"path": paths, "prediction": preds})
    pred_col = cols.get("prediction") or cols.get("label")
    out = pd.DataFrame({"path": df[path_col].astype(str)})
    out["path"] = out["path"].str.replace("\\", "/", regex=False)
    out["path"] = out["path"].map(lambda s: s if s.endswith("/") else s + "/")
    out["prediction"] = df[pred_col] if pred_col else pd.NA
    return out


def validate_submission(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"submission missing columns {missing}")
    out = df[list(SUBMISSION_COLUMNS)].copy()
    out["path"] = out["path"].astype(str).str.replace("\\", "/", regex=False)
    out["path"] = out["path"].map(lambda s: s if s.endswith("/") else s + "/")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    if out["prediction"].isna().any():
        raise ValueError("submission contains non-numeric predictions")
    out["prediction"] = out["prediction"].astype(int)
    if (out["prediction"] < 0).any() or (out["prediction"] >= NUM_CLASSES).any():
        raise ValueError("predictions must be integers in [0, 39]")
    if len(out) != NUM_TEST_CLIPS:
        raise ValueError(f"expected {NUM_TEST_CLIPS} rows, got {len(out)}")
    if out["path"].nunique() != NUM_TEST_CLIPS:
        raise ValueError("duplicate or missing test paths")
    return out.reset_index(drop=True)


def write_submission(df: pd.DataFrame, out_path: Path) -> Path:
    clean = validate_submission(df)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    return out_path
