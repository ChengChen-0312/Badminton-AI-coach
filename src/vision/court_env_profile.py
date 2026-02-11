from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse


def _normalize_env_mapping(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = str(k or "").strip()
        if not key or not key.startswith("BADC_"):
            continue
        if v is None:
            continue
        out[key] = str(v).strip()
    return out


def parse_env_file(path: str | Path) -> Dict[str, str]:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    out: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.startswith("BADC_"):
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        out[key] = value
    return out


def _resolve_profile_path(raw: str, base_dir: Optional[Path]) -> Path:
    cand = Path(str(raw)).expanduser()
    if cand.is_absolute():
        return cand
    if base_dir is not None:
        return (base_dir / cand).resolve()
    return cand.resolve()


def resolve_court_env_overrides(
    vision_cfg: Mapping[str, Any] | None,
    *,
    base_dir: Optional[Path] = None,
) -> Tuple[Dict[str, str], Optional[str]]:
    cfg = vision_cfg if isinstance(vision_cfg, Mapping) else {}
    merged: Dict[str, str] = {}
    profile_path: Optional[str] = None

    profile_raw = cfg.get("court_env_file")
    if isinstance(profile_raw, str) and profile_raw.strip():
        path = _resolve_profile_path(profile_raw.strip(), base_dir=base_dir)
        if path.exists() and path.is_file():
            merged.update(parse_env_file(path))
            profile_path = str(path)

    merged.update(_normalize_env_mapping(cfg.get("court_env")))
    return merged, profile_path


def apply_court_env_overrides(overrides: Mapping[str, Any] | None) -> Dict[str, str]:
    normalized = _normalize_env_mapping(overrides)
    applied: Dict[str, str] = {}
    for k, v in normalized.items():
        os.environ[k] = v
        applied[k] = v
    return applied


def resolve_and_apply_court_env_overrides(
    vision_cfg: Mapping[str, Any] | None,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    overrides, profile_path = resolve_court_env_overrides(vision_cfg, base_dir=base_dir)
    applied = apply_court_env_overrides(overrides)
    return {
        "profile_path": profile_path,
        "num_vars": int(len(applied)),
        "applied": applied,
    }


def resolve_local_path_from_url_or_path(raw: str, repo_root: Path) -> Optional[Path]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    parsed = urlparse(txt)
    if parsed.scheme in ("http", "https"):
        return None
    if parsed.scheme == "file":
        p = Path(parsed.path).expanduser()
        return p
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p
