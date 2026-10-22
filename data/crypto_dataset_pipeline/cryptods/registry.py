"""Feature registry: one record per output column (formula, units, clock, availability, status)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

STATUSES = ("implemented_verified", "implemented_untested_real", "unavailable_from_current_source",
            "not_implemented")


@dataclass
class FeatureSpec:
    name: str
    family: str
    formula: str
    unit: str
    clock: str
    inputs: str
    provider: str
    availability: str
    nature: str = "observed"            # observed | estimated | constructed | test_statistic
    windows: str = ""
    warmup: str = ""
    staleness: str = ""
    null_reasons: list = field(default_factory=list)
    strict_eligible: bool = True
    variant: str = ""
    reference: str = ""
    library_diff: str = ""
    status: str = "implemented_untested_real"
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def to_markdown(specs: list[FeatureSpec]) -> str:
    lines = ["| column | family | unit | clock | nature | strict | status |", "|---|---|---|---|---|---|---|"]
    for s in specs:
        lines.append(f"| `{s.name}` | {s.family} | {s.unit} | {s.clock} | {s.nature} | "
                     f"{'yes' if s.strict_eligible else 'no'} | {s.status} |")
    return "\n".join(lines)
