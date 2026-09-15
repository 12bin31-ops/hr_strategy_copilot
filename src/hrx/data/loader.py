"""합성 데이터 로더. 없으면 즉석에서 생성한다(오프라인 재현성)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from hrx.config import SETTINGS
from hrx.data.generate import generate

TABLES = ("org_units", "employees", "key_positions", "succession", "voc", "business_plan")


@dataclass
class HRData:
    org_units: pd.DataFrame
    employees: pd.DataFrame
    key_positions: pd.DataFrame
    succession: pd.DataFrame
    voc: pd.DataFrame
    business_plan: pd.DataFrame


def ensure_data(data_dir: Path | None = None) -> Path:
    d = data_dir or SETTINGS.paths.synthetic
    missing = [t for t in TABLES if not (d / f"{t}.csv").exists()]
    if missing:
        generate(d)
    return d


def load(data_dir: Path | None = None) -> HRData:
    d = ensure_data(data_dir)
    frames = {t: pd.read_csv(d / f"{t}.csv") for t in TABLES}
    return HRData(**frames)


@lru_cache(maxsize=1)
def cached() -> HRData:
    return load()
