from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from har.constants import IMU_FEATS_PER_SENSOR, NUM_CLASSES, NUM_IMU_SENSORS, NUM_SKELETON_JOINTS
from har.loaders import append_skeleton_velocity, append_temporal_diff, normalize_pose_torch


class ResBlock3d(nn.Module):
    def __init__(self, ch: int, stride: tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch)
        self.down = None
        if stride != (1, 1, 1):
            self.down = nn.Sequential(
                nn.Conv3d(ch, ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        skip = x if self.down is None else self.down(x)
        return F.relu(h + skip, inplace=True)


def width_channels(base: int, width: float) -> int:
    return max(8, int(round(base * width / 8.0) * 8))


class Conv3dEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 512, width: float = 1.0):
        super().__init__()
        c1 = width_channels(48, width)
        c2 = width_channels(96, width)
        c3 = width_channels(192, width)
        c4 = width_channels(256, width)
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, c1, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.layer1 = nn.Sequential(ResBlock3d(c1), ResBlock3d(c1))
        self.down1 = nn.Conv3d(c1, c2, 1, stride=(1, 2, 2), bias=False)
        self.layer2 = nn.Sequential(ResBlock3d(c2), ResBlock3d(c2))
        self.down2 = nn.Conv3d(c2, c3, 1, stride=(1, 2, 2), bias=False)
        self.layer3 = nn.Sequential(ResBlock3d(c3), ResBlock3d(c3))
        self.down3 = nn.Conv3d(c3, c4, 1, stride=(2, 1, 1), bias=False)
        self.layer4 = nn.Sequential(ResBlock3d(c4))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(c4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(self.down1(h))
        h = self.layer3(self.down2(h))
        h = self.layer4(self.down3(h))
        return self.proj(self.pool(h).flatten(1))


class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, state = self.gru(h)
        cat = torch.cat([state[0], state[1]], dim=1)
        return self.proj(cat)


class MultiModalHAR(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feat_dim: int = 256,
        width: float = 1.0,
        depth_motion: bool = False,
        skel_velocity: bool = False,
        skel_norm: bool = False,
    ):
        super().__init__()
        self.width = float(width)
        self.depth_motion = bool(depth_motion)
        self.skel_velocity = bool(skel_velocity)
        self.skel_norm = bool(skel_norm)
        in_ch = 6 if self.depth_motion else 3
        self.depth_enc = Conv3dEncoder(in_ch, feat_dim, width=self.width)
        self.imu_enc = SeqEncoder(NUM_IMU_SENSORS * IMU_FEATS_PER_SENSOR, 96, 256)
        skel_in = NUM_SKELETON_JOINTS * (6 if self.skel_velocity else 3)
        self.skel_enc = SeqEncoder(skel_in, 96, 256)
        self.depth_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        self.skel_delta = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes))
        nn.init.zeros_(self.skel_delta[-1].weight)
        nn.init.zeros_(self.skel_delta[-1].bias)
        self.imu_delta = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes))
        nn.init.zeros_(self.imu_delta[-1].weight)
        nn.init.zeros_(self.imu_delta[-1].bias)
        self.ir_enc = Conv3dEncoder(3, feat_dim, width=max(0.5, self.width * 0.5))
        self.ir_delta = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, num_classes))
        nn.init.zeros_(self.ir_delta[-1].weight)
        nn.init.zeros_(self.ir_delta[-1].bias)

    def forward(self, batch: dict) -> torch.Tensor:
        depth = batch["depth"]
        if self.depth_motion:
            depth = append_temporal_diff(depth)
        depth_f = self.depth_enc(depth)
        logits = self.depth_head(depth_f)
        mask = batch["mask"]
        use_ir = mask[:, 1].reshape(-1, 1)
        use_imu = mask[:, 3].reshape(-1, 1)
        use_skel = mask[:, 4].reshape(-1, 1)
        if float(use_skel.sum()) > 0:
            skel = batch["skeleton"]
            if self.skel_norm:
                skel = normalize_pose_torch(skel)
            if self.skel_velocity:
                skel = append_skeleton_velocity(skel)
            skel_f = self.skel_enc(skel.flatten(2)) * use_skel
            logits = logits + self.skel_delta(skel_f)
        if float(use_ir.sum()) > 0:
            ir_f = self.ir_enc(batch["ir"]) * use_ir
            logits = logits + self.ir_delta(ir_f)
        if float(use_imu.sum()) > 0:
            imu = batch["imu"]
            mean = imu.mean(dim=1, keepdim=True)
            std = imu.std(dim=1, keepdim=True).clamp_min(1e-6)
            imu_f = self.imu_enc((imu - mean) / std) * use_imu
            logits = logits + self.imu_delta(imu_f)
        return logits


def model_from_train_cfg(cfg: dict | None, num_classes: int = NUM_CLASSES) -> MultiModalHAR:
    cfg = cfg or {}
    return MultiModalHAR(
        num_classes=num_classes,
        width=float(cfg.get("width", 1.0)),
        depth_motion=bool(cfg.get("depth_motion", False)),
        skel_velocity=bool(cfg.get("skel_velocity", False)),
        skel_norm=bool(cfg.get("skel_norm", False)),
    )


def adapt_state_depth_motion(model: MultiModalHAR, state: dict) -> dict:
    key = "depth_enc.stem.0.weight"
    if key not in state:
        return state
    old = state[key]
    new_shape = tuple(model.state_dict()[key].shape)
    if tuple(old.shape) == new_shape:
        return state
    if old.shape[1] == 3 and new_shape[1] == 6:
        expanded = model.state_dict()[key].clone()
        expanded[:, :3] = old
        expanded[:, 3:] = 0
        state = dict(state)
        state[key] = expanded
        print("expanded depth stem 3ch -> 6ch (motion channels zero-init)")
    return state


def adapt_state_skel_velocity(model: MultiModalHAR, state: dict) -> dict:
    key = "skel_enc.conv.0.weight"
    if key not in state:
        return state
    old = state[key]
    new_shape = tuple(model.state_dict()[key].shape)
    if tuple(old.shape) == new_shape:
        return state
    if old.shape[1] == NUM_SKELETON_JOINTS * 3 and new_shape[1] == NUM_SKELETON_JOINTS * 6:
        expanded = model.state_dict()[key].clone()
        expanded[:, : old.shape[1]] = old
        expanded[:, old.shape[1] :] = 0
        state = dict(state)
        state[key] = expanded
        print("expanded skel conv 51 -> 102 (velocity channels zero-init)")
    return state


def adapt_init_state(model: MultiModalHAR, state: dict) -> dict:
    state = adapt_state_depth_motion(model, state)
    return adapt_state_skel_velocity(model, state)


def time_window_starts(num_frames: int, win: int) -> list[int]:
    if num_frames <= win:
        return [0]
    starts = list(range(0, num_frames - win + 1, win))
    if starts[-1] != num_frames - win:
        starts.append(num_frames - win)
    return starts


def _slice_video(batch: dict, start: int, win: int) -> dict:
    out = dict(batch)
    for key in ("depth", "ir", "thermal"):
        x = out.get(key)
        if torch.is_tensor(x) and x.dim() >= 4:
            out[key] = x.narrow(-3, start, win)
    return out


def _flip_views(batch: dict) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    for key in ("depth", "ir", "thermal"):
        if torch.is_tensor(out.get(key)):
            out[key] = torch.flip(out[key], dims=[-1])
    return out


def _crop_resize_hw(
    x: torch.Tensor, top: int, left: int, nh: int, nw: int, out_h: int, out_w: int
) -> torch.Tensor:
    if x.dim() == 4:
        # (C, T, H, W)
        y = x[:, :, top : top + nh, left : left + nw].permute(1, 0, 2, 3)
        y = torch.nn.functional.interpolate(
            y, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        return y.permute(1, 0, 2, 3)
    if x.dim() == 5:
        # (B, C, T, H, W)
        b, c, t, _, _ = x.shape
        y = x[:, :, :, top : top + nh, left : left + nw].permute(0, 2, 1, 3, 4)
        y = y.reshape(b * t, c, nh, nw)
        y = torch.nn.functional.interpolate(
            y, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        return y.reshape(b, t, c, out_h, out_w).permute(0, 2, 1, 3, 4)
    raise ValueError(f"unexpected video shape {tuple(x.shape)}")


def _center_crop_resize(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale >= 0.999:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    top = (h - nh) // 2
    left = (w - nw) // 2
    return _crop_resize_hw(x, top, left, nh, nw, h, w)


def _corner_crop_resize(x: torch.Tensor, scale: float, corner: str) -> torch.Tensor:
    if scale >= 0.999:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    if corner == "tl":
        top, left = 0, 0
    elif corner == "tr":
        top, left = 0, w - nw
    elif corner == "bl":
        top, left = h - nh, 0
    else:
        top, left = h - nh, w - nw
    return _crop_resize_hw(x, top, left, nh, nw, h, w)


def _spatial_views(batch: dict, tta_crop: bool) -> list[dict]:
    if not tta_crop:
        return [batch]
    views = [batch]
    for scale in (0.875, 0.75):
        cropped = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        for key in ("depth", "ir", "thermal"):
            if torch.is_tensor(cropped.get(key)):
                cropped[key] = _center_crop_resize(cropped[key], scale)
        views.append(cropped)
    for corner in ("tl", "tr", "bl", "br"):
        cropped = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        for key in ("depth", "ir", "thermal"):
            if torch.is_tensor(cropped.get(key)):
                cropped[key] = _corner_crop_resize(cropped[key], 0.875, corner)
        views.append(cropped)
    return views


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def prepare_adabn(model: nn.Module) -> None:
    """Reset BN running stats; use cumulative average over the calibration pass."""
    model.eval()
    n = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None
            module.train()
            n += 1
    print(f"adabn prepared bn_layers={n}")


def adabn_forward(model: nn.Module, batch: dict, device: torch.device) -> None:
    """One unlabeled forward to update BN running stats (no logits used)."""
    with torch.no_grad():
        moved = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
            if k in {"depth", "ir", "thermal", "imu", "skeleton", "radar", "mask"}
        }
        if moved["depth"].dim() == 4:
            moved = {k: v.unsqueeze(0) for k, v in moved.items()}
        model(moved)


def finish_adabn(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def predict_logits(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        moved = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
            if k in {"depth", "ir", "thermal", "imu", "skeleton", "radar", "mask"}
        }
        if moved["depth"].dim() == 4:
            moved = {k: v.unsqueeze(0) for k, v in moved.items()}
        return model(moved)


def _average_window_logits(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    tta_flip: bool,
    win_frames: int,
    tta_crop: bool = False,
) -> torch.Tensor:
    depth = batch["depth"]
    t_len = int(depth.shape[-3])
    total = None
    n = 0
    for spatial in _spatial_views(batch, tta_crop):
        for start in time_window_starts(t_len, win_frames):
            view = _slice_video(spatial, start, min(win_frames, t_len))
            part = predict_logits(model, view, device)
            total = part if total is None else total + part
            n += 1
            if tta_flip:
                part = predict_logits(model, _flip_views(view), device)
                total = total + part
                n += 1
    return total / max(n, 1)


def predict_label(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    tta_flip: bool = False,
    win_frames: int = 8,
    tta_crop: bool = False,
) -> int:
    logits = _average_window_logits(
        model, batch, device, tta_flip, win_frames, tta_crop=tta_crop
    )
    return int(logits.argmax(dim=1).item())


def predict_label_ensemble(
    models: list[nn.Module],
    batch: dict,
    device: torch.device,
    tta_flip: bool = False,
    win_frames: int = 8,
    tta_crop: bool = False,
) -> int:
    logits = None
    for model in models:
        part = _average_window_logits(
            model, batch, device, tta_flip, win_frames, tta_crop=tta_crop
        )
        logits = part if logits is None else logits + part
    return int(logits.argmax(dim=1).item())
