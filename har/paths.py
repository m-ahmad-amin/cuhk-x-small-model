from __future__ import annotations

from pathlib import Path

from har.constants import MODALITY_ALIASES


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_test_csv(data_root: Path) -> Path | None:
    root = Path(data_root)
    names = ("test.csv", "sample_submission.csv")
    rels = [
        Path("."),
        Path("test_file"),
        Path("Testing") / "test_file",
        Path("Testing"),
        Path(names[0]).parent,
    ]
    candidates: list[Path] = []
    for rel in rels:
        for name in names:
            candidates.append(root / rel / name)
    csvs = [p for p in candidates if p.is_file()]
    for p in csvs:
        if p.name == "test.csv":
            return p
    return csvs[0] if csvs else None


def find_test_root(data_root: Path) -> Path | None:
    root = Path(data_root)
    candidates = [
        root / "small_model_track_test",
        root / "Testing" / "data" / "small_model_track_test",
        root / "Testing" / "small_model_track_test",
        root,
    ]
    for cand in candidates:
        if "__macosx" in cand.as_posix().lower():
            continue
        if cand.is_dir() and any(p.is_dir() and p.name.startswith("SM_test_") for p in cand.glob("SM_test_*")):
            return cand
    return None


def find_har_root(data_root: Path) -> Path | None:
    root = Path(data_root)
    candidates = [
        root / "HAR" / "data",
        root / "Training" / "data" / "HAR" / "data",
        root / "data" / "HAR" / "data",
        root / "HAR",
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    return None


def find_modality_dir(clip_dir: Path, modality: str) -> Path | None:
    if not clip_dir.is_dir():
        return None
    wanted = {name.lower() for name in MODALITY_ALIASES[modality]}
    for child in clip_dir.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child
    if modality in {"depth", "ir", "thermal"}:
        from har.constants import IMAGE_EXTS

        if any(p.suffix.lower() in IMAGE_EXTS for p in clip_dir.iterdir() if p.is_file()):
            return clip_dir
    return None


def clip_id_from_path(path_str: str) -> str:
    text = path_str.replace("\\", "/").rstrip("/")
    return text.split("/")[-1]


def sibling_modality_dir(trial_dir: Path, modality: str) -> Path | None:
    trial_dir = Path(trial_dir)
    try:
        user_dir = trial_dir.parent
        action_dir = user_dir.parent
        mod_dir = action_dir.parent
        har_data = mod_dir.parent
        rel = Path(action_dir.name) / user_dir.name / trial_dir.name
    except Exception:
        return None
    wanted = {n.lower() for n in MODALITY_ALIASES[modality]}
    if har_data.is_dir():
        for child in har_data.iterdir():
            if child.is_dir() and child.name.lower() in wanted:
                cand = child / rel
                if cand.is_dir():
                    return cand
    if mod_dir.is_dir() and mod_dir.name.lower() in wanted:
        return trial_dir
    return None


def modality_trial_dir(har_root: Path, path_key: str, modality: str) -> Path | None:
    rel = path_key.replace("\\", "/").strip("/")
    if not har_root.is_dir():
        return None
    for name in MODALITY_ALIASES[modality]:
        cand = har_root / name / rel
        if cand.is_dir():
            return cand
    return None


def har_has_modality(har_root: Path, modality: str) -> bool:
    wanted = {n.lower() for n in MODALITY_ALIASES[modality]}
    if not har_root.is_dir():
        return False
    for child in har_root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return any(child.rglob("*"))
    return False
