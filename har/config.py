from __future__ import annotations

from pathlib import Path

DRIVE_CODE = Path("/content/drive/MyDrive/cuhk-x-small-model")
DRIVE_DATA = Path("/content/drive/MyDrive/Small-Model-Track")
COLAB_EXTRACT = Path("/content/har_extract")
COLAB_TEST = Path("/content/small_model_track_test")
COLAB_WORK = Path("/content/cuhk-x-small-model")

TRAIN_ZIP_DIR = DRIVE_DATA / "Training" / "data"
TEST_CSV = DRIVE_DATA / "Testing" / "test.csv"
TEST_ZIP = DRIVE_DATA / "Testing" / "data" / "small_model_track_test.zip"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir(kind: str) -> Path:
    drive = DRIVE_CODE / kind
    if DRIVE_CODE.exists():
        drive.mkdir(parents=True, exist_ok=True)
        return drive
    local = repo_root() / kind
    local.mkdir(parents=True, exist_ok=True)
    return local
