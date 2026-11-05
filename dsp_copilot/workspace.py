"""Per-session state: the signals, filters and figures the agent is working with.

Tools refer to objects by short names, so large arrays never pass through the LLM.
"""
from __future__ import annotations

import re
from pathlib import Path

from .dsp import FilterSpec
from .signals import Signal


class ToolError(ValueError):
    """A problem the LLM can correct (bad name, bad argument). Returned to it as text."""


class Workspace:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.signals: dict[str, Signal] = {}
        self.filters: dict[str, FilterSpec] = {}
        self._figure_count = 0

    # ---------------------------------------------------------------- signals
    def add_signal(self, signal: Signal) -> str:
        """Store a signal under a unique name and return that name."""
        base, name, n = signal.name, signal.name, 2
        while name in self.signals:
            name, n = f"{base}_{n}", n + 1
        signal.name = name
        self.signals[name] = signal
        return name

    def get_signal(self, name: str) -> Signal:
        if name not in self.signals:
            raise ToolError(f"Unknown signal '{name}'. Available signals: {sorted(self.signals)}")
        return self.signals[name]

    # ---------------------------------------------------------------- filters
    def add_filter(self, spec: FilterSpec) -> str:
        filter_id = f"filter_{len(self.filters) + 1}"
        self.filters[filter_id] = spec
        return filter_id

    def get_filter(self, filter_id: str) -> FilterSpec:
        if filter_id not in self.filters:
            raise ToolError(f"Unknown filter '{filter_id}'. Available filters: {sorted(self.filters)}")
        return self.filters[filter_id]

    # ---------------------------------------------------------------- figures
    def new_figure_path(self, stem: str) -> Path:
        self._figure_count += 1
        safe = re.sub(r"[^a-z0-9_]+", "_", stem.lower())
        return self.out_dir / f"{self._figure_count:02d}_{safe}.png"

    # ------------------------------------------------------------------ prompt
    def describe(self) -> list[dict]:
        """Signal list for the LLM's context."""
        return [s.summary() for s in self.signals.values()]
