from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Equation:
    """A named residual equation.

    Residuals are considered satisfied when |residual|/scale is small.
    """

    name: str
    residual: float
    scale: float = 1.0
