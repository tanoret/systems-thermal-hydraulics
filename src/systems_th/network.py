from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .connection import Connection
from .props import WaterIAPWS, WaterProps
from .solver import newton_solve, SolveOptions, SolveResult
from .components.base import Component


@dataclass
class Network:
    """A network of components connected by directed Connections."""

    props: WaterProps = field(default_factory=WaterIAPWS)
    components: List[Component] = field(default_factory=list)
    connections: Dict[str, Connection] = field(default_factory=dict)

    def add_component(self, comp: Component) -> None:
        self.components.append(comp)

    def add_connection(self, conn: Connection) -> None:
        if conn.name in self.connections:
            raise KeyError(f"Connection '{conn.name}' already exists")
        self.connections[conn.name] = conn

    def connect(
        self,
        src: Component,
        src_port: str,
        dst: Component,
        dst_port: str,
        name: str,
        m_guess: float = 100.0,
        p_guess: float = 1e6,
        h_guess: float = 1e6,
    ) -> Connection:
        conn = Connection.create(name=name, m_guess=m_guess, p_guess=p_guess, h_guess=h_guess)
        self.add_connection(conn)
        src.connect_outlet(src_port, conn)
        dst.connect_inlet(dst_port, conn)
        return conn

    def all_variables(self) -> List:
        vars_ = []
        for c in self.connections.values():
            vars_.extend(c.variables())
        for comp in self.components:
            vars_.extend(comp.variables())
        return vars_

    def free_variables(self) -> List:
        return [v for v in self.all_variables() if not v.fixed]

    def residuals(self) -> List:
        eqs = []
        for comp in self.components:
            eqs.extend(comp.equations(self.props))
        return eqs

    def solve(self, options: Optional[SolveOptions] = None) -> SolveResult:
        if options is None:
            options = SolveOptions()
        return newton_solve(self, options)

    def summary(self) -> str:
        lines = []
        lines.append(f"Network with {len(self.components)} components and {len(self.connections)} connections")
        lines.append("Connections:")
        for name, c in self.connections.items():
            lines.append(
                f"  - {name}: m={c.m.value:.4g}{' (fixed)' if c.m.fixed else ''}, "\
                f"p={c.p.value:.4g}{' (fixed)' if c.p.fixed else ''}, "\
                f"h={c.h.value:.4g}{' (fixed)' if c.h.fixed else ''}"
            )
        return "\n".join(lines)
