from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelId(str, Enum):
    ORNITH = "ornith-35b"
    GEMMA = "gemma-4-e2b"


class LeaseState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class QueuedRequest:
    model_id: ModelId
    request_id: str
    sequence: int = 0
