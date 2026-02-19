from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Variable:
    """A scalar model variable.

    Parameters
    ----------
    name:
        Unique name for debugging and reporting.
    value:
        Current value.
    fixed:
        If True, variable is excluded from the solver unknown vector.
    lower, upper:
        Optional bounds. The solver clips trial points into bounds.
    """

    name: str
    value: float
    fixed: bool = False
    lower: Optional[float] = None
    upper: Optional[float] = None

    def fix(self, value: Optional[float] = None) -> None:
        if value is not None:
            self.value = float(value)
        self.fixed = True

    def unfix(self) -> None:
        self.fixed = False

    def clip(self) -> None:
        if self.lower is not None and self.value < self.lower:
            self.value = self.lower
        if self.upper is not None and self.value > self.upper:
            self.value = self.upper
