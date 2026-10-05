from __future__ import annotations

import numpy as np

from har.constants import IMU_FEATS_PER_SENSOR, NUM_IMU_SENSORS, NUM_SKELETON_JOINTS
from har.dataset import load_multimodal_from_test_clip, load_multimodal_from_train_trial

L_SHO, R_SHO = 5, 6
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x))


def _dominant_hz(signal: np.ndarray, fs: float = 50.0) -> float:
    if signal.size < 8:
        return 0.0
    x = signal - signal.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spec[0] = 0.0
    if spec.max() < 1e-6:
        return 0.0
    return float(freqs[int(np.argmax(spec))])


def extract_features(sample: dict) -> dict[str, float]:
    imu = np.asarray(sample["imu"])
    skel = np.asarray(sample["skeleton"])
    mask = np.asarray(sample["mask"])

    t = imu.shape[0]
    sensors = imu.reshape(t, NUM_IMU_SENSORS, IMU_FEATS_PER_SENSOR)
    acc = np.linalg.norm(sensors[:, :, :3], axis=2)
    gyro = np.linalg.norm(sensors[:, :, 3:6], axis=2)
    active = acc.std(axis=0) > 1e-5
    acc_mag = acc[:, active].mean(axis=1) if active.any() else acc.mean(axis=1)
    gyro_mag = gyro[:, active].mean(axis=1) if active.any() else gyro.mean(axis=1)
    leg = acc[:, 3:5].mean(axis=1) if acc.shape[1] >= 5 else acc_mag

    joints = skel.reshape(skel.shape[0], NUM_SKELETON_JOINTS, -1)
    hip = (joints[:, L_HIP, :2] + joints[:, R_HIP, :2]) * 0.5
    sho = (joints[:, L_SHO, :2] + joints[:, R_SHO, :2]) * 0.5
    torso = sho - hip
    torso_len = np.linalg.norm(torso, axis=1) + 1e-6
    upright = np.abs(torso[:, 1]) / torso_len

    wrist = (joints[:, L_WRI, :2] + joints[:, R_WRI, :2]) * 0.5
    ankle = (joints[:, L_ANK, :2] + joints[:, R_ANK, :2]) * 0.5
    wrist_speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1) if wrist.shape[0] > 1 else np.zeros(1)
    ankle_speed = np.linalg.norm(np.diff(ankle, axis=0), axis=1) if ankle.shape[0] > 1 else np.zeros(1)
    hip_vert = hip[:, 1]
    arm_span = np.linalg.norm(joints[:, L_WRI, :2] - joints[:, R_WRI, :2], axis=1)

    return {
        "has_imu": float(mask[3]),
        "has_skel": float(mask[4]),
        "acc_std": _safe_std(acc_mag),
        "acc_mean": float(acc_mag.mean()) if acc_mag.size else 0.0,
        "gyro_std": _safe_std(gyro_mag),
        "step_hz": _dominant_hz(acc_mag),
        "leg_std": _safe_std(leg),
        "upright": float(np.median(upright)) if upright.size else 0.0,
        "hip_osc": _safe_std(hip_vert),
        "wrist_speed": float(wrist_speed.mean()) if wrist_speed.size else 0.0,
        "ankle_speed": float(ankle_speed.mean()) if ankle_speed.size else 0.0,
        "arm_span_std": _safe_std(arm_span),
        "energy": _safe_std(acc_mag) + 0.15 * _safe_std(gyro_mag),
    }


def features_to_label(feat: dict[str, float]) -> int:
    energy = feat["energy"]
    hz = feat["step_hz"]
    upright = feat["upright"]
    ankle = feat["ankle_speed"]
    wrist = feat["wrist_speed"]
    hip_osc = feat["hip_osc"]
    arm_span_std = feat["arm_span_std"]
    leg_std = feat["leg_std"]

    if upright < 0.35 and energy < 0.35 and ankle < 0.02:
        return 33
    if energy > 0.9 and hz >= 2.2 and upright > 0.45:
        return 28
    if energy > 0.7 and arm_span_std > 0.08 and upright > 0.45:
        return 30
    if hz >= 1.2 and hz < 2.4 and energy > 0.25 and upright > 0.45:
        return 36
    if hip_osc > 0.04 and ankle < 0.05 and energy > 0.2 and upright > 0.4:
        return 29
    if energy < 0.25 and ankle < 0.02 and upright > 0.25:
        if wrist > 0.01:
            return 24
        return 34
    if energy < 0.18 and ankle < 0.015 and upright > 0.55:
        return 32

    if energy > 0.3 or leg_std > 0.2:
        return 36
    return 24


def predict_clip(clip_dir, train_layout: bool = False) -> int:
    loader = load_multimodal_from_train_trial if train_layout else load_multimodal_from_test_clip
    sample = loader(clip_dir)
    if float(sample["mask"].sum()) == 0:
        return 36
    return features_to_label(extract_features(sample))
