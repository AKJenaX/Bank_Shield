from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseTask:
    name: str
    dataset_file: str
