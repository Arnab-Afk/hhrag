from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    stages: list[tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append((name, (time.perf_counter() - start) * 1000.0))

    def get(self, name: str) -> float:
        for stage_name, ms in self.stages:
            if stage_name == name:
                return ms
        return 0.0

    def as_list(self) -> list[dict[str, float | str]]:
        return [{"name": name, "ms": round(ms, 3)} for name, ms in self.stages]

    @property
    def total_ms(self) -> float:
        return sum(ms for _, ms in self.stages)
