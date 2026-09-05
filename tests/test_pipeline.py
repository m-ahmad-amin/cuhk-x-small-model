from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har.constants import NUM_CLASSES, NUM_SKELETON_JOINTS, NUM_TEST_CLIPS
from har.dataset import load_multimodal_from_test_clip, restrict_sample_modalities, to_tensors
from har.heuristic import features_to_label, extract_features, predict_clip
from har.loaders import load_imu, person_crop_box, normalize_pose
from har.model import MultiModalHAR, model_size_mb, predict_label
from har.submit import default_rows, validate_submission, write_submission


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)


def _write_imu(path: Path, acc_z: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(acc_z.size, dtype=np.float32)
    table = np.stack(
        [
            np.zeros_like(acc_z),
            np.zeros_like(acc_z),
            acc_z,
            np.zeros_like(acc_z) + 0.05,
            np.zeros_like(acc_z),
            acc_z * 0.2,
            np.zeros_like(acc_z),
            np.zeros_like(acc_z),
            t * 0.0,
        ],
        axis=1,
    )
    header = "accX,accY,accZ,gyroX,gyroY,gyroZ,angleX,angleY,angleZ"
    np.savetxt(path, table, delimiter=",", header=header, comments="")


def _write_skeleton(path: Path, upright: bool, ankle_motion: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = 32
    joints = np.zeros((t, NUM_SKELETON_JOINTS, 3), dtype=np.float32)
    joints[:, 11] = 0.0
    joints[:, 12] = [0.2, 0.0, 0.0]
    if upright:
        joints[:, 5] = [0.0, 1.0, 0.0]
        joints[:, 6] = [0.2, 1.0, 0.0]
    else:
        joints[:, 5] = [1.0, 0.0, 0.0]
        joints[:, 6] = [1.2, 0.0, 0.0]
    phase = np.linspace(0, 4 * np.pi, t)
    joints[:, 15, 0] = ankle_motion * np.sin(phase)
    joints[:, 16, 0] = ankle_motion * np.sin(phase + np.pi)
    np.save(path, joints)


def make_clip(root: Path, clip_id: str, kind: str) -> Path:
    clip = root / "small_model_track_test" / clip_id
    fs, n = 50.0, 128
    t = np.arange(n) / fs
    if kind == "walk":
        acc = 1.0 + 0.6 * np.sin(2 * np.pi * 1.7 * t)
        _write_imu(clip / "IMU" / "s1.csv", acc)
        _write_skeleton(clip / "Skeleton" / "pose.npy", upright=True, ankle_motion=0.15)
    elif kind == "run":
        acc = 1.0 + 2.2 * np.sin(2 * np.pi * 2.8 * t)
        _write_imu(clip / "IMU" / "s1.csv", acc)
        _write_skeleton(clip / "Skeleton" / "pose.npy", upright=True, ankle_motion=0.25)
    elif kind == "lie":
        acc = 0.02 * np.ones(n)
        _write_imu(clip / "IMU" / "s1.csv", acc)
        _write_skeleton(clip / "Skeleton" / "pose.npy", upright=False, ankle_motion=0.0)
    else:
        raise ValueError(kind)
    _write_png(clip / "Depth_Color" / "Depth_0001_Color.png")
    return clip


def test_heuristic_on_synthetic_clips() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mapping = {"walk": 36, "run": 28, "lie": 33}
        for kind, expected in mapping.items():
            clip = make_clip(root, f"SM_test_{kind}", kind)
            pred = predict_clip(clip)
            feat = extract_features(load_multimodal_from_test_clip(clip))
            assert pred == expected, f"{kind}: got {pred} expected {expected} feats={feat}"
            assert features_to_label(feat) == expected


def test_skel_residual_zero_at_init() -> None:
    import torch

    model = MultiModalHAR()
    model.eval()
    depth = torch.zeros(1, 3, 8, 64, 64)
    skel = torch.zeros(1, 32, 17, 3)
    mask = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    batch = {
        "depth": depth,
        "ir": depth,
        "thermal": depth,
        "imu": torch.zeros(1, 128, 45),
        "skeleton": skel,
        "radar": torch.zeros(1, 32, 8),
        "mask": mask,
    }
    a = model(batch)
    batch["mask"] = torch.tensor([[1.0, 0, 0, 0, 1.0, 0]])
    b = model(batch)
    assert torch.allclose(a, b, atol=1e-5)


def test_model_forward_and_size() -> None:
    import torch

    model = MultiModalHAR()
    size = model_size_mb(model)
    assert size < 100.0, size
    with tempfile.TemporaryDirectory() as tmp:
        clip = make_clip(Path(tmp), "SM_test_0001", "walk")
        sample = to_tensors(load_multimodal_from_test_clip(clip))
        label = predict_label(model, sample, torch.device("cpu"))
        assert 0 <= label < NUM_CLASSES


def test_submission_format() -> None:
    df = default_rows()
    clean = validate_submission(df)
    assert len(clean) == NUM_TEST_CLIPS
    assert list(clean.columns) == ["path", "prediction"]
    assert clean.loc[0, "path"] == "small_model_track_test/SM_test_0001/"
    assert clean.loc[NUM_TEST_CLIPS - 1, "path"] == "small_model_track_test/SM_test_0405/"
    with tempfile.TemporaryDirectory() as tmp:
        out = write_submission(clean, Path(tmp) / "submission.csv")
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["path"].endswith("/")
        assert len(rows) == NUM_TEST_CLIPS


def test_chinese_imu_devices() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "up(LA+RA+C).csv"
        path.write_text(
            "时间,设备名称,加速度X(g),加速度Y(g),加速度Z(g),"
            "角速度X(°/s),角速度Y(°/s),角速度Z(°/s),角度X(°),角度Y(°),角度Z(°)\n"
            "t,WTLA(xx),1,0,0,0,0,0,0,0,0\n"
            "t,WTRA(xx),2,0,0,0,0,0,0,0,0\n"
            "t,WTC(xx),3,0,0,0,0,0,0,0,0\n",
            encoding="utf-8",
        )
        arr, ok = load_imu(Path(tmp), length=8)
        assert ok
        assert abs(float(arr[0, 0]) - 1) < 1e-3
        assert abs(float(arr[0, 9]) - 2) < 1e-3
        assert abs(float(arr[0, 18]) - 3) < 1e-3


def test_restrict_modalities_zeros_imu() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        clip = make_clip(Path(tmp), "SM_test_0001", "walk")
        sample = load_multimodal_from_test_clip(clip)
        assert float(sample["mask"][3]) == 1.0
        sample = restrict_sample_modalities(sample, ["depth"])
        assert float(sample["mask"][0]) == 1.0
        assert float(sample["mask"][3]) == 0.0
        assert float(sample["mask"][4]) == 0.0


def test_normalize_pose_centers_hips() -> None:
    seq = np.zeros((4, NUM_SKELETON_JOINTS, 3), dtype=np.float32)
    seq[:, 11] = [10.0, 20.0, 0.0]
    seq[:, 12] = [12.0, 20.0, 0.0]
    seq[:, 5] = [10.0, 24.0, 0.0]
    seq[:, 6] = [12.0, 24.0, 0.0]
    out = normalize_pose(seq)
    hip = 0.5 * (out[:, 11] + out[:, 12])
    assert np.allclose(hip, 0.0, atol=1e-5)
    sho = 0.5 * (out[:, 5] + out[:, 6])
    assert abs(float(np.linalg.norm(sho[0])) - 1.0) < 1e-4


def test_person_crop_finds_blob() -> None:
    frames = []
    for i in range(4):
        img = np.zeros((80, 120, 3), dtype=np.float32)
        img[20:50, 10 + i : 40 + i] = 1.0
        frames.append(img)
    x0, y0, x1, y1 = person_crop_box(frames)
    assert x1 - x0 < 120
    assert y1 - y0 < 80
    assert x0 < 40
    assert y0 < 30


def main() -> None:
    test_heuristic_on_synthetic_clips()
    print("ok  heuristic walk/jog/lie")
    test_restrict_modalities_zeros_imu()
    print("ok  depth-only mask zeros IMU/skeleton")
    test_normalize_pose_centers_hips()
    print("ok  pose normalize")
    test_person_crop_finds_blob()
    print("ok  person crop finds moving blob")
    test_chinese_imu_devices()
    print("ok  Chinese IMU device split")
    test_skel_residual_zero_at_init()
    print("ok  skeleton residual is zero at init")
    test_model_forward_and_size()
    print("ok  model forward + size < 100 MB")
    test_submission_format()
    print("ok  submission format")


if __name__ == "__main__":
    main()
