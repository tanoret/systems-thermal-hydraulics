from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..connection import Connection
from ..equation import Equation
from ..variable import Variable


@dataclass
class Component:
    """Base class for all components.

    Components expose *ports* (inlets/outlets) and contribute residual equations.
    """

    name: str
    inlets: Dict[str, Connection] = field(default_factory=dict)
    outlets: Dict[str, Connection] = field(default_factory=dict)

    def connect_inlet(self, port: str, conn: Connection) -> None:
        self.inlets[port] = conn

    def connect_outlet(self, port: str, conn: Connection) -> None:
        self.outlets[port] = conn

    def variables(self) -> List[Variable]:
        # Override to include internal variables.
        return []

    def equations(self, props) -> List[Equation]:
        # Override in subclasses.
        raise NotImplementedError

    def _req_in(self, port: str) -> Connection:
        if port not in self.inlets:
            raise KeyError(f"{self.name}: inlet '{port}' not connected")
        return self.inlets[port]

    def _req_out(self, port: str) -> Connection:
        if port not in self.outlets:
            raise KeyError(f"{self.name}: outlet '{port}' not connected")
        return self.outlets[port]
