from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)

    if is_dataclass(obj):
        obj = asdict(obj)

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def env_summary() -> dict[str, str]:
    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
