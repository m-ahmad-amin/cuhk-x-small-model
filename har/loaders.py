from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from har.constants import (
    ARRAY_EXTS,
    IMAGE_EXTS,
    IMU_FEATS_PER_SENSOR,
    IMU_LENGTH,
    IMAGE_SIZE,
    NUM_FRAMES,
    NUM_IMU_SENSORS,
    NUM_SKELETON_JOINTS,
    RADAR_LENGTH,
    SKELETON_LENGTH,
    TABLE_EXTS,
)


def _sorted_files(folder: Path, exts: set[str]) -> list[Path]:
    files = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parts_l = {part.lower() for part in p.parts}
        if p.name.startswith(".") or "__macosx" in parts_l or "visualizations" in parts_l:
            continue
        files.append(p)
    return sorted(files, key=lambda p: p.as_posix().lower())


def resample_time(x: np.ndarray, length: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros((length,) + x.shape[1:], dtype=np.float32)
    if x.shape[0] == 1:
        return np.repeat(x.astype(np.float32), length, axis=0)
    src = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    out = np.empty((length,) + x.shape[1:], dtype=np.float32)
    flat = x.reshape(x.shape[0], -1).astype(np.float32)
    for c in range(flat.shape[1]):
        out.reshape(length, -1)[:, c] = np.interp(dst, src, flat[:, c])
    return out


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((k,), dtype=int)
    if n == 1:
        return np.zeros((k,), dtype=int)
    return np.linspace(0, n - 1, num=k).round().astype(int)


def _read_image_native(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        pass
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return None
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[-1] == 4:
            bgr = bgr[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    except Exception:
        return None


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _read_image_rgb(path: Path, size: int) -> np.ndarray | None:
    arr = _read_image_native(path)
    if arr is None:
        return None
    return _resize_rgb(arr, size)


def person_crop_box(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    grays = []
    for frame in frames:
        if frame.ndim == 3:
            grays.append(frame.mean(axis=2).astype(np.float32))
        else:
            grays.append(frame.astype(np.float32))
    h, w = grays[0].shape
    stack = np.stack(grays, axis=0)
    motion = np.abs(np.diff(stack, axis=0)).mean(axis=0) if len(grays) > 1 else np.zeros((h, w), np.float32)
    var = stack.var(axis=0)
    mean = stack.mean(axis=0)
    gy = np.abs(np.diff(mean, axis=0, prepend=mean[:1]))
    gx = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
    heat = motion + var + 0.25 * (gx + gy)
    if float(heat.max()) < 1e-6:
        return 0, 0, w, h
    heat = heat / (float(heat.max()) + 1e-8)
    thr = max(0.12, float(np.quantile(heat, 0.82)))
    ys, xs = np.where(heat >= thr)
    if ys.size < 16:
        return 0, 0, w, h
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(0.18 * (y1 - y0) + 0.02 * h)
    pad_x = int(0.18 * (x1 - x0) + 0.02 * w)
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(h, y1 + pad_y)
    x1 = min(w, x1 + pad_x)
    if (y1 - y0) < 0.2 * h or (x1 - x0) < 0.15 * w:
        return 0, 0, w, h
    return x0, y0, x1, y1


def apply_person_crop(frames: list[np.ndarray]) -> list[np.ndarray]:
    x0, y0, x1, y1 = person_crop_box(frames)
    if x0 == 0 and y0 == 0 and x1 == frames[0].shape[1] and y1 == frames[0].shape[0]:
        return frames
    return [frame[y0:y1, x0:x1] for frame in frames]


def normalize_clip(video: np.ndarray) -> np.ndarray:
    x = video.astype(np.float32, copy=False)
    std = float(x.std())
    if std < 1e-6:
        return x
    return (x - float(x.mean())) / (std + 1e-6)


def append_temporal_diff(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    diff = x[:, :, 1:] - x[:, :, :-1]
    pad = torch.zeros_like(x[:, :, :1])
    out = torch.cat([x, torch.cat([pad, diff], dim=2)], dim=1)
    return out.squeeze(0) if squeeze else out


def append_skeleton_velocity(skel: torch.Tensor) -> torch.Tensor:
    if skel.dim() == 3:
        skel = skel.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    vel = skel[:, 1:] - skel[:, :-1]
    pad = torch.zeros_like(skel[:, :1])
    out = torch.cat([skel, torch.cat([pad, vel], dim=1)], dim=-1)
    return out.squeeze(0) if squeeze else out


def normalize_pose_torch(skel: torch.Tensor) -> torch.Tensor:
    if skel.dim() == 3:
        skel = skel.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if skel.shape[2] < 13:
        return skel.squeeze(0) if squeeze else skel
    x = skel.clone()
    hip = 0.5 * (x[:, :, 11, :3] + x[:, :, 12, :3])
    x[:, :, :, :3] = x[:, :, :, :3] - hip.unsqueeze(2)
    sho = 0.5 * (x[:, :, 5, :3] + x[:, :, 6, :3])
    torso = torch.linalg.norm(sho, dim=-1)
    scale = torso.median(dim=1).values.clamp_min(1e-6).view(-1, 1, 1, 1)
    x[:, :, :, :3] = x[:, :, :, :3] / scale
    return x.squeeze(0) if squeeze else x


def augment_depth_clip(depth: torch.Tensor) -> torch.Tensor:
    """depth: (C, T, H, W). Random crop/scale, temporal speed, mild intensity."""
    c, t, h, w = depth.shape
    x = depth
    if torch.rand(()) < 0.8:
        scale = float(torch.empty(1).uniform_(0.72, 1.0).item())
        nh = max(8, int(round(h * scale)))
        nw = max(8, int(round(w * scale)))
        top = int(torch.randint(0, h - nh + 1, (1,)).item()) if nh < h else 0
        left = int(torch.randint(0, w - nw + 1, (1,)).item()) if nw < w else 0
        crop = x[:, :, top : top + nh, left : left + nw]
        x = torch.nn.functional.interpolate(
            crop.permute(1, 0, 2, 3),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    if t >= 4 and torch.rand(()) < 0.5:
        speed = float(torch.empty(1).uniform_(0.8, 1.25).item())
        new_t = max(4, int(round(t / speed)))
        y = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(new_t, h, w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        if new_t >= t:
            start = int(torch.randint(0, new_t - t + 1, (1,)).item()) if new_t > t else 0
            x = y[:, start : start + t]
        else:
            pad = t - new_t
            left = pad // 2
            x = torch.nn.functional.pad(y, (0, 0, 0, 0, left, pad - left), mode="replicate")
    if torch.rand(()) < 0.5:
        gain = float(torch.empty(1).uniform_(0.85, 1.15).item())
        bias = float(torch.empty(1).uniform_(-0.1, 0.1).item())
        x = x * gain + bias
    return x


def load_frames(
    folder: Path | None,
    num_frames: int = NUM_FRAMES,
    size: int = IMAGE_SIZE,
    crop_person: bool = False,
) -> tuple[np.ndarray, bool]:
    empty = np.zeros((3, num_frames, size, size), dtype=np.float32)
    if folder is None or not folder.is_dir():
        return empty, False
    files = _sorted_files(folder, IMAGE_EXTS)
    if not files:
        return empty, False
    order = [files[i] for i in _uniform_indices(len(files), min(num_frames, len(files)))]
    leftover = [p for p in files if p not in order]
    clips = []
    reader = _read_image_native if crop_person else (lambda p: _read_image_rgb(p, size))
    for path in order + leftover:
        arr = reader(path)
        if arr is None:
            continue
        clips.append(arr)
        if len(clips) >= num_frames:
            break
    if not clips:
        return empty, False
    while len(clips) < num_frames:
        clips.append(clips[-1])
    clips = clips[:num_frames]
    if crop_person:
        clips = apply_person_crop(clips)
        clips = [_resize_rgb(frame, size) for frame in clips]
    video = np.transpose(np.stack(clips, axis=0), (3, 0, 1, 2))
    return normalize_clip(video), True


def _read_table(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".npy"}:
        return np.asarray(np.load(path), dtype=np.float32)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.size == 0 or np.isnan(raw[0]).all():
        raw = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32, skip_header=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
    keep_cols = ~np.isnan(raw).all(axis=0)
    raw = raw[:, keep_cols]
    keep_rows = ~np.isnan(raw).all(axis=1)
    raw = raw[keep_rows]
    return np.nan_to_num(raw, nan=0.0).astype(np.float32)


def _read_csv_df(path: Path):
    import pandas as pd

    return pd.read_csv(path, encoding="utf-8-sig")


def _find_col(df, *needles: str) -> str | None:
    needles_l = [n.lower() for n in needles]
    for col in df.columns:
        name = str(col).lower().replace(" ", "")
        if all(n.replace(" ", "") in name for n in needles_l):
            return col
    return None


def _imu_nine(df) -> np.ndarray:
    acc = [_find_col(df, "加速度", ax) or _find_col(df, "acc", ax) for ax in ("x", "y", "z")]
    gyro = [
        _find_col(df, "角速度", ax) or _find_col(df, "gyro", ax) or _find_col(df, "angular", ax)
        for ax in ("x", "y", "z")
    ]
    ang = [_find_col(df, "角度", ax) or _find_col(df, "angle", ax) for ax in ("x", "y", "z")]
    cols = acc + gyro + ang
    if all(c is not None for c in cols):
        return df.loc[:, cols].to_numpy(dtype=np.float32)
    numeric = df.select_dtypes(include=[np.number])
    arr = numeric.to_numpy(dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, IMU_FEATS_PER_SENSOR), dtype=np.float32)
    if arr.shape[1] < IMU_FEATS_PER_SENSOR:
        arr = np.pad(arr, ((0, 0), (0, IMU_FEATS_PER_SENSOR - arr.shape[1])))
    return arr[:, :IMU_FEATS_PER_SENSOR]


_DEVICE_INDEX = {
    "WTLA": 0,
    "WTRA": 1,
    "WTC": 2,
    "WTLL": 3,
    "WTRL": 4,
}


def _device_index(name: str) -> int | None:
    text = str(name).upper()
    for key in ("WTLA", "WTRA", "WTLL", "WTRL", "WTC"):
        if key in text:
            return _DEVICE_INDEX[key]
    return None


def load_imu(folder: Path | None, length: int = IMU_LENGTH) -> tuple[np.ndarray, bool]:
    dim = NUM_IMU_SENSORS * IMU_FEATS_PER_SENSOR
    empty = np.zeros((length, dim), dtype=np.float32)
    if folder is None or not folder.is_dir():
        return empty, False
    files = _sorted_files(folder, TABLE_EXTS | {".npy"})
    if not files:
        return empty, False

    buckets: list[list[np.ndarray]] = [[] for _ in range(NUM_IMU_SENSORS)]
    unlabeled: list[np.ndarray] = []
    for path in files:
        if path.suffix.lower() == ".npy":
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            unlabeled.append(arr)
            continue
        try:
            df = _read_csv_df(path)
        except Exception:
            arr = _read_table(path)
            if arr.size:
                unlabeled.append(arr)
            continue
        if df.empty:
            continue
        dev_col = _find_col(df, "设备") or _find_col(df, "device")
        if dev_col is not None:
            for dev_name, group in df.groupby(dev_col, sort=False):
                idx = _device_index(dev_name)
                feats = _imu_nine(group)
                if feats.size == 0:
                    continue
                if idx is None:
                    unlabeled.append(feats)
                else:
                    buckets[idx].append(feats)
        else:
            unlabeled.append(_imu_nine(df))

    streams: list[np.ndarray] = []
    any_data = False
    u = 0
    for i in range(NUM_IMU_SENSORS):
        parts = buckets[i]
        if not parts and u < len(unlabeled):
            parts = [unlabeled[u]]
            u += 1
        if parts:
            any_data = True
            cat = np.concatenate(parts, axis=0)
            if cat.shape[1] < IMU_FEATS_PER_SENSOR:
                cat = np.pad(cat, ((0, 0), (0, IMU_FEATS_PER_SENSOR - cat.shape[1])))
            streams.append(resample_time(cat[:, :IMU_FEATS_PER_SENSOR], length))
        else:
            streams.append(np.zeros((length, IMU_FEATS_PER_SENSOR), dtype=np.float32))
    if not any_data:
        return empty, False
    return np.concatenate(streams, axis=1).astype(np.float32), True


def _joints_from_obj(obj) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    if isinstance(obj, dict):
        for key in ("keypoints", "joints", "pose", "skeleton", "frames"):
            if key in obj:
                return _joints_from_obj(obj[key])
        if "people" in obj:
            return _joints_from_obj(obj["people"])
    if isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            frames = []
            for item in obj:
                kps = item.get("keypoints") or item.get("joints") or item.get("pose")
                if kps is None:
                    continue
                frames.append(np.asarray(kps, dtype=np.float32).reshape(-1))
            if frames:
                return np.stack(frames, axis=0)
        arr = np.asarray(obj, dtype=np.float32)
        if arr.size:
            return arr
    return None


def _reshape_skeleton(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim == 4:
        x = x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim == 3:
        if x.shape[1] == NUM_SKELETON_JOINTS:
            pass
        elif x.shape[0] == NUM_SKELETON_JOINTS:
            x = np.transpose(x, (1, 0, 2))
        elif x.shape[2] == NUM_SKELETON_JOINTS:
            x = np.transpose(x, (0, 2, 1))
        else:
            t, a, b = x.shape
            if a * b % NUM_SKELETON_JOINTS == 0:
                x = x.reshape(t, NUM_SKELETON_JOINTS, -1)
    if x.ndim == 2:
        t, f = x.shape
        if f % NUM_SKELETON_JOINTS == 0:
            x = x.reshape(t, NUM_SKELETON_JOINTS, f // NUM_SKELETON_JOINTS)
        else:
            need = NUM_SKELETON_JOINTS * 3
            if f < need:
                x = np.pad(x, ((0, 0), (0, need - f)))
            x = x[:, :need].reshape(t, NUM_SKELETON_JOINTS, 3)
    if x.ndim != 3:
        x = np.zeros((1, NUM_SKELETON_JOINTS, 3), dtype=np.float32)
    t, j, c = x.shape
    if j < NUM_SKELETON_JOINTS:
        x = np.pad(x, ((0, 0), (0, NUM_SKELETON_JOINTS - j), (0, 0)))
    x = x[:, :NUM_SKELETON_JOINTS]
    if c < 3:
        x = np.pad(x, ((0, 0), (0, 0), (0, 3 - c)))
    return x[:, :, :4].astype(np.float32)


def normalize_pose(seq: np.ndarray) -> np.ndarray:
    x = np.asarray(seq, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] < 13:
        return x
    hip = 0.5 * (x[:, 11, :3] + x[:, 12, :3])
    x = x.copy()
    x[:, :, :3] = x[:, :, :3] - hip[:, None, :]
    sho = 0.5 * (x[:, 5, :3] + x[:, 6, :3])
    torso = np.linalg.norm(sho, axis=1)
    scale = float(np.median(torso)) if torso.size else 1.0
    if scale < 1e-6:
        scale = 1.0
    x[:, :, :3] = x[:, :, :3] / scale
    return x


def load_skeleton(folder: Path | None, length: int = SKELETON_LENGTH) -> tuple[np.ndarray, bool]:
    empty = np.zeros((length, NUM_SKELETON_JOINTS, 3), dtype=np.float32)
    if folder is None or not folder.is_dir():
        return empty, False
    files = _sorted_files(folder, ARRAY_EXTS | TABLE_EXTS)
    if not files:
        return empty, False
    chunks: list[np.ndarray] = []
    for path in files:
        try:
            if path.suffix.lower() == ".npy":
                obj = np.load(path, allow_pickle=True)
            elif path.suffix.lower() == ".npz":
                npz = np.load(path, allow_pickle=True)
                obj = npz[npz.files[0]]
            elif path.suffix.lower() == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() in {".pkl", ".pickle"}:
                obj = pickle.loads(path.read_bytes())
            else:
                obj = _read_table(path)
            arr = _joints_from_obj(obj)
            if arr is None:
                continue
            chunks.append(_reshape_skeleton(arr)[..., :3])
        except Exception:
            continue
    if not chunks:
        return empty, False
    seq = np.concatenate(chunks, axis=0)
    seq = resample_time(seq, length)
    return seq.astype(np.float32), True


def load_radar(folder: Path | None, length: int = RADAR_LENGTH) -> tuple[np.ndarray, bool]:
    empty = np.zeros((length, 8), dtype=np.float32)
    if folder is None or not folder.is_dir():
        return empty, False
    files = _sorted_files(folder, TABLE_EXTS | {".npy"})
    if not files:
        return empty, False
    try:
        df = _read_csv_df(files[0])
    except Exception:
        arr = _read_table(files[0])
        if arr.size == 0:
            return empty, False
        df = None
    else:
        if df.empty:
            return empty, False

    stats = []
    if df is not None and {"x", "y", "z"}.issubset({c.lower() for c in df.columns}):
        cols = {c.lower(): c for c in df.columns}
        frame_col = cols.get("frame")
        groups = df.groupby(frame_col, sort=True) if frame_col else [(0, df)]
        for _, pts in groups:
            xyz = pts[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=np.float32)
            v = pts[cols["v"]].to_numpy(dtype=np.float32) if "v" in cols else np.zeros(len(pts))
            snr = pts[cols["snr"]].to_numpy(dtype=np.float32) if "snr" in cols else np.zeros(len(pts))
            stats.append(
                np.array(
                    [
                        len(pts),
                        xyz[:, 0].mean(),
                        xyz[:, 1].mean(),
                        xyz[:, 2].mean(),
                        xyz.std(),
                        float(v.mean()) if v.size else 0.0,
                        float(snr.mean()) if snr.size else 0.0,
                        np.linalg.norm(xyz, axis=1).mean(),
                    ],
                    dtype=np.float32,
                )
            )
    else:
        arr = _read_table(files[0])
        if arr.size == 0:
            return empty, False
        if arr.shape[1] < 8:
            arr = np.pad(arr, ((0, 0), (0, 8 - arr.shape[1])))
        stats = list(arr[:, :8])
    if not stats:
        return empty, False
    seq = resample_time(np.stack(stats, axis=0), length)
    return seq.astype(np.float32), True
