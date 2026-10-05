from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from har.constants import (
    IMU_FEATS_PER_SENSOR,
    IMU_LENGTH,
    MASK_MODALITIES,
    NUM_FRAMES,
    NUM_IMU_SENSORS,
    NUM_SKELETON_JOINTS,
    RADAR_LENGTH,
    SKELETON_LENGTH,
    TRAIN_USERS,
    IMAGE_SIZE,
)
from har.loaders import (
    augment_depth_clip,
    load_frames,
    load_imu,
    load_radar,
    load_skeleton,
    normalize_clip,
)
from har.paths import find_modality_dir, sibling_modality_dir

_USER_RE = re.compile(r"(\d+)")
_ACTION_RE = re.compile(r"^(\d+)_")


@dataclass
class ClipRef:
    clip_dir: Path
    label: int | None
    user_id: int | None
    path_key: str


def parse_action_id(folder_name: str) -> int | None:
    match = _ACTION_RE.match(folder_name)
    if match:
        return int(match.group(1))
    if folder_name.isdigit():
        return int(folder_name)
    return None


def parse_user_id(folder_name: str) -> int | None:
    match = _USER_RE.search(folder_name)
    return int(match.group(1)) if match else None


def index_train_clips(har_root: Path, users: tuple[int, ...] = TRAIN_USERS) -> list[ClipRef]:
    har_root = Path(har_root)
    grouped: dict[tuple[str, str, str], Path] = {}
    modality_dirs = [p for p in har_root.iterdir() if p.is_dir()]
    modality_dirs.sort(key=lambda p: (0 if "depth" in p.name.lower() else 1, p.name))
    for mod_dir in modality_dirs:
        for action_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            for user_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
                user_id = parse_user_id(user_dir.name)
                if user_id is not None and user_id not in users:
                    continue
                for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                    key = (action_dir.name, user_dir.name, trial_dir.name)
                    if key not in grouped:
                        grouped[key] = trial_dir
    clips: list[ClipRef] = []
    for (action, user, trial), trial_dir in grouped.items():
        label = parse_action_id(action)
        if label is None:
            continue
        clips.append(
            ClipRef(
                clip_dir=trial_dir,
                label=int(label),
                user_id=parse_user_id(user),
                path_key=f"{action}/{user}/{trial}/",
            )
        )
    return clips


def restrict_sample_modalities(sample: dict, modalities: list[str] | tuple[str, ...]) -> dict:
    allowed = {m.lower() for m in modalities}
    mask = np.array(sample["mask"], dtype=np.float32, copy=True)
    for i, name in enumerate(MASK_MODALITIES):
        if name not in allowed:
            mask[i] = 0.0
    sample["mask"] = mask
    return sample


def load_multimodal_from_test_clip(
    clip_dir: Path,
    crop_person: bool = False,
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> dict:
    clip_dir = Path(clip_dir)
    depth, d_ok = load_frames(
        find_modality_dir(clip_dir, "depth"),
        crop_person=crop_person,
        num_frames=num_frames,
        size=image_size,
    )
    ir, i_ok = load_frames(
        find_modality_dir(clip_dir, "ir"),
        crop_person=crop_person,
        num_frames=num_frames,
        size=image_size,
    )
    thermal, t_ok = load_frames(
        find_modality_dir(clip_dir, "thermal"),
        crop_person=crop_person,
        num_frames=num_frames,
        size=image_size,
    )
    imu, u_ok = load_imu(find_modality_dir(clip_dir, "imu"))
    skeleton, s_ok = load_skeleton(find_modality_dir(clip_dir, "skeleton"))
    radar, r_ok = load_radar(find_modality_dir(clip_dir, "radar"))
    return {
        "depth": depth,
        "ir": ir,
        "thermal": thermal,
        "imu": imu,
        "skeleton": skeleton,
        "radar": radar,
        "mask": np.array([d_ok, i_ok, t_ok, u_ok, s_ok, r_ok], dtype=np.float32),
    }


def load_multimodal_from_train_trial(
    trial_dir: Path,
    crop_person: bool = False,
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> dict:
    trial_dir = Path(trial_dir)
    if sibling_modality_dir(trial_dir, "depth") is None and not trial_dir.is_dir():
        return load_multimodal_from_test_clip(
            trial_dir, crop_person=crop_person, num_frames=num_frames, image_size=image_size
        )

    def sibling(mod_key: str) -> Path | None:
        return sibling_modality_dir(trial_dir, mod_key)

    depth, d_ok = load_frames(
        sibling("depth"), crop_person=crop_person, num_frames=num_frames, size=image_size
    )
    ir, i_ok = load_frames(
        sibling("ir"), crop_person=crop_person, num_frames=num_frames, size=image_size
    )
    thermal, t_ok = load_frames(
        sibling("thermal"), crop_person=crop_person, num_frames=num_frames, size=image_size
    )
    imu, u_ok = load_imu(sibling("imu"))
    skeleton, s_ok = load_skeleton(sibling("skeleton"))
    radar, r_ok = load_radar(sibling("radar"))
    return {
        "depth": depth,
        "ir": ir,
        "thermal": thermal,
        "imu": imu,
        "skeleton": skeleton,
        "radar": radar,
        "mask": np.array([d_ok, i_ok, t_ok, u_ok, s_ok, r_ok], dtype=np.float32),
    }


def empty_sample(num_frames: int = NUM_FRAMES, image_size: int = IMAGE_SIZE) -> dict:
    return {
        "depth": np.zeros((3, num_frames, image_size, image_size), dtype=np.float32),
        "ir": np.zeros((3, num_frames, image_size, image_size), dtype=np.float32),
        "thermal": np.zeros((3, num_frames, image_size, image_size), dtype=np.float32),
        "imu": np.zeros((IMU_LENGTH, NUM_IMU_SENSORS * IMU_FEATS_PER_SENSOR), dtype=np.float32),
        "skeleton": np.zeros((SKELETON_LENGTH, NUM_SKELETON_JOINTS, 3), dtype=np.float32),
        "radar": np.zeros((RADAR_LENGTH, 8), dtype=np.float32),
        "mask": np.zeros((6,), dtype=np.float32),
    }


def to_tensors(sample: dict) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()}


def infer_cached_num_frames(index_csv: Path) -> int:
    npy = Path(pd.read_csv(index_csv).iloc[0]["npy"])
    return int(np.load(npy).shape[1])


def infer_cached_image_size(index_csv: Path) -> int:
    npy = Path(pd.read_csv(index_csv).iloc[0]["npy"])
    return int(np.load(npy).shape[2])


class HARClipDataset(Dataset):
    def __init__(
        self,
        clips: list[ClipRef],
        train_layout: bool = True,
        augment: bool = False,
        crop_person: bool = False,
        num_frames: int = NUM_FRAMES,
        image_size: int = IMAGE_SIZE,
    ):
        self.clips = clips
        self.train_layout = train_layout
        self.augment = augment
        self.crop_person = crop_person
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        ref = self.clips[idx]
        if self.train_layout:
            sample = to_tensors(
                load_multimodal_from_train_trial(
                    ref.clip_dir,
                    crop_person=self.crop_person,
                    num_frames=self.num_frames,
                    image_size=self.image_size,
                )
            )
        else:
            sample = to_tensors(
                load_multimodal_from_test_clip(
                    ref.clip_dir,
                    crop_person=self.crop_person,
                    num_frames=self.num_frames,
                    image_size=self.image_size,
                )
            )
        if self.augment:
            sample["depth"] = augment_depth_clip(sample["depth"])
            if torch.rand(()) < 0.5:
                sample["depth"] = torch.flip(sample["depth"], dims=[-1])
        sample["label"] = torch.tensor(-1 if ref.label is None else ref.label, dtype=torch.long)
        sample["path_key"] = ref.path_key
        return sample


class CachedDepthDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        augment: bool = False,
        use_imu: bool = True,
        use_skeleton: bool = False,
        use_ir: bool = False,
    ):
        self.table = pd.read_csv(index_csv)
        self.augment = augment
        self.use_imu = use_imu
        self.use_skeleton = use_skeleton
        self.use_ir = use_ir
        self.num_frames = infer_cached_num_frames(index_csv)
        self.image_size = infer_cached_image_size(index_csv)

    def modalities(self) -> list[str]:
        mods = ["depth"]
        if self.use_imu and "imu_npy" in self.table.columns:
            filled = self.table["imu_npy"].fillna("").astype(str).str.strip()
            if bool((filled != "").any()):
                mods.append("imu")
        if self.use_skeleton and "skel_npy" in self.table.columns:
            filled = self.table["skel_npy"].fillna("").astype(str).str.strip()
            if bool((filled != "").any()):
                mods.append("skeleton")
        if self.use_ir and "ir_npy" in self.table.columns:
            filled = self.table["ir_npy"].fillna("").astype(str).str.strip()
            if bool((filled != "").any()):
                mods.append("ir")
        return mods

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        row = self.table.iloc[idx]
        sample = empty_sample(num_frames=self.num_frames, image_size=self.image_size)
        depth = normalize_clip(np.load(row["npy"]).astype(np.float32))
        sample["depth"] = depth
        sample["mask"][0] = 1.0
        tensors = to_tensors(sample)
        ir_path = row["ir_npy"] if self.use_ir and "ir_npy" in self.table.columns else None
        if ir_path is not None and pd.notna(ir_path) and str(ir_path).strip():
            ir_file = Path(str(ir_path))
            if ir_file.is_file():
                tensors["ir"] = torch.from_numpy(
                    normalize_clip(np.load(ir_file).astype(np.float32))
                )
                tensors["mask"][1] = 1.0
        if self.augment:
            tensors["depth"] = augment_depth_clip(tensors["depth"])
            if tensors["mask"][1] > 0:
                tensors["ir"] = augment_depth_clip(tensors["ir"])
            if torch.rand(()) < 0.5:
                tensors["depth"] = torch.flip(tensors["depth"], dims=[-1])
                tensors["ir"] = torch.flip(tensors["ir"], dims=[-1])
        tensors["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        tensors["path_key"] = str(row["path_key"])
        tensors["user_id"] = int(row["user_id"]) if pd.notna(row["user_id"]) else -1
        imu_path = row["imu_npy"] if self.use_imu and "imu_npy" in self.table.columns else None
        if imu_path is not None and pd.notna(imu_path) and str(imu_path).strip():
            imu_file = Path(str(imu_path))
            if imu_file.is_file():
                tensors["imu"] = torch.from_numpy(np.load(imu_file).astype(np.float32))
                tensors["mask"][3] = 1.0
        skel_path = row["skel_npy"] if self.use_skeleton and "skel_npy" in self.table.columns else None
        if skel_path is not None and pd.notna(skel_path) and str(skel_path).strip():
            skel_file = Path(str(skel_path))
            if skel_file.is_file():
                tensors["skeleton"] = torch.from_numpy(np.load(skel_file).astype(np.float32))
                tensors["mask"][4] = 1.0
        return tensors


def collate_clips(batch: list[dict]) -> dict:
    keys = ("depth", "ir", "thermal", "imu", "skeleton", "radar", "mask", "label")
    out = {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}
    out["path_key"] = [item["path_key"] for item in batch]
    return out
