"""Inspectable strategy specifications.

A :class:`StrategySpec` documents everything a caller needs to know before
running a strategy: required input columns (including timestamp rules), the
indicators and features its rules use, its parameters, warm-up behaviour,
invalid-data handling and stateful behaviour.  Specs are plain frozen
dataclasses so they can be printed, compared and turned into DataFrames.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def param(default: Any, doc: str) -> Any:
    """Declare a config field together with its documentation."""
    return field(default=default, metadata={"doc": doc})


@dataclass(frozen=True)
class ColumnRequirement:
    """One input column of the bar DataFrame."""

    name: str
    required: bool
    role: str  # timestamp | ohlc | volume | validity | segmentation | reference
    description: str
    fallback: str | None = None  # behaviour when an optional column is absent


@dataclass(frozen=True)
class IndicatorSpec:
    """An indicator the rules depend on, with its exact variant."""

    name: str
    description: str
    initialization: str
    segment_handling: str
    parameters: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureSpec:
    """A value captured at decision time for a candidate event.

    ``required`` means the strategy's rules use it.  Optional features are only
    for event analysis / modeling and can be skipped with ``features="required"``.
    Every feature is evaluated on the signal (decision) bar only, using
    information available at that bar's decision time.
    """

    name: str
    required: bool
    description: str
    dtype: str = "float64"
    formula: str = ""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: Any
    type: str
    description: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    title: str
    description: str
    source: str
    aliases: tuple[str, ...]
    inputs: tuple[ColumnRequirement, ...]
    timestamp_rules: tuple[str, ...]
    indicators: tuple[IndicatorSpec, ...]
    features: tuple[FeatureSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    warmup: str
    invalid_data: tuple[str, ...]
    entry_rules: tuple[str, ...]
    exit_rules: tuple[str, ...]
    stateful_behavior: tuple[str, ...]
    execution: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    # -- convenience accessors -------------------------------------------------
    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.inputs if c.required)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.inputs if not c.required)

    @property
    def required_features(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.required)

    @property
    def optional_features(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if not f.required)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def inputs_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(c) for c in self.inputs])

    def indicators_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(i) for i in self.indicators])

    def features_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(f) for f in self.features])

    def parameters_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(p) for p in self.parameters])

    def describe(self) -> str:
        """Human-readable multi-line summary."""
        def block(title: str, rows) -> list[str]:
            return [f"{title}:"] + [f"  - {r}" for r in rows]

        lines = [f"{self.name} - {self.title}", self.description, f"source: {self.source}"]
        if self.aliases:
            lines.append("aliases: " + ", ".join(self.aliases))
        lines += block("inputs", [
            f"{c.name} ({'required' if c.required else 'optional'}, {c.role}): {c.description}"
            + (f" [absent -> {c.fallback}]" if c.fallback else "") for c in self.inputs])
        lines += block("timestamp rules", self.timestamp_rules)
        lines += block("indicators", [f"{i.name}: {i.description} init={i.initialization}; "
                                      f"segments={i.segment_handling}" for i in self.indicators])
        lines += block("entry rules", self.entry_rules)
        lines += block("exit rules", self.exit_rules)
        lines += block("stateful behavior", self.stateful_behavior)
        lines += block("execution", self.execution)
        lines += block("features", [f"{f.name} ({'required' if f.required else 'optional'}): "
                                    f"{f.description}" for f in self.features])
        lines += block("parameters", [f"{p.name}={p.default!r}: {p.description}" for p in self.parameters])
        lines.append(f"warm-up: {self.warmup}")
        lines += block("invalid data", self.invalid_data)
        if self.limitations:
            lines += block("limitations", self.limitations)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.describe()


def parameters_from_config(config_class: type) -> tuple[ParameterSpec, ...]:
    """Build :class:`ParameterSpec` rows from a config dataclass' fields."""
    out = []
    for f in dataclasses.fields(config_class):
        default = f.default if f.default is not dataclasses.MISSING else None
        out.append(ParameterSpec(f.name, default, getattr(f.type, "__name__", str(f.type)),
                                 f.metadata.get("doc", "")))
    return tuple(out)
