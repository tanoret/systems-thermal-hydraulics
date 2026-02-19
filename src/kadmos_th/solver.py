from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .equation import Equation


@dataclass
class SolveOptions:
    max_iter: int = 60
    tol: float = 1e-7          # scaled residual norm
    xtol: float = 1e-9         # variable step norm
    fd_eps: float = 1e-6       # relative finite diff step
    damping: bool = True
    verbose: bool = True
    print_worst: int = 5       # show worst residuals when verbose


@dataclass
class SolveResult:
    converged: bool
    iterations: int
    residual_norm: float
    message: str


def _pack_vars(vars_) -> np.ndarray:
    return np.array([v.value for v in vars_], dtype=float)


def _unpack_vars(vars_, x: np.ndarray) -> None:
    for v, val in zip(vars_, x):
        v.value = float(val)
        v.clip()


def _scaled_residual_vector(eqs: List[Equation]) -> np.ndarray:
    return np.array([eq.residual / (eq.scale if eq.scale != 0 else 1.0) for eq in eqs], dtype=float)


def _fd_jacobian(network, free_vars, f0: np.ndarray, fd_eps: float) -> np.ndarray:
    n = len(free_vars)
    m = len(f0)
    J = np.zeros((m, n), dtype=float)

    x0 = _pack_vars(free_vars)

    for j in range(n):
        x = x0.copy()
        step = fd_eps * max(1.0, abs(x[j]))
        x[j] += step
        _unpack_vars(free_vars, x)

        eqs = network.residuals()
        f1 = _scaled_residual_vector(eqs)
        J[:, j] = (f1 - f0) / step

    _unpack_vars(free_vars, x0)
    return J


def _worst_residuals(eqs: List[Equation], k: int) -> list[tuple[str, float]]:
    vals = [(eq.name, abs(eq.residual / (eq.scale if eq.scale != 0 else 1.0))) for eq in eqs]
    vals.sort(key=lambda t: t[1], reverse=True)
    return vals[:k]


def newton_solve(network, options: SolveOptions) -> SolveResult:
    free_vars = network.free_variables()
    if options.verbose:
        print(f"[systems-th] Unknowns: {len(free_vars)} (free variables)")

    if len(free_vars) == 0:
        eqs = network.residuals()
        f = _scaled_residual_vector(eqs)
        nrm = float(np.linalg.norm(f, ord=2))
        return SolveResult(converged=nrm < options.tol, iterations=0, residual_norm=nrm, message="No free variables")


    for it in range(1, options.max_iter + 1):
        eqs0 = network.residuals()
        f0 = _scaled_residual_vector(eqs0)
        nrm0 = float(np.linalg.norm(f0, ord=2))

        if options.verbose:
            msg = f"[systems-th] iter {it:02d}: |F|={nrm0:.3e} eqs={len(eqs0)}"
            print(msg)
            if it == 1 or it % 10 == 0:
                for name, val in _worst_residuals(eqs0, options.print_worst):
                    print(f"    worst: {name} -> {val:.3e}")

        if nrm0 < options.tol:
            return SolveResult(True, it - 1, nrm0, "Converged (residual norm)")

        J = _fd_jacobian(network, free_vars, f0, options.fd_eps)

        rhs = -f0
        try:
            dx, *_ = np.linalg.lstsq(J, rhs, rcond=None)
        except Exception as e:
            return SolveResult(False, it, nrm0, f"Linear solve failed: {e}")

        step_norm = float(np.linalg.norm(dx, ord=2))
        if step_norm < options.xtol:
            return SolveResult(True, it, nrm0, "Converged (step norm)")

        x0 = _pack_vars(free_vars)

        alpha = 1.0
        if options.damping:
            improved = False
            for _ in range(14):
                x_trial = x0 + alpha * dx
                _unpack_vars(free_vars, x_trial)
                f_trial = _scaled_residual_vector(network.residuals())
                nrm_trial = float(np.linalg.norm(f_trial, ord=2))
                if nrm_trial <= nrm0:
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                _unpack_vars(free_vars, x0)
                return SolveResult(False, it, nrm0, "Damping failed to improve residual")
        else:
            x_trial = x0 + dx
            _unpack_vars(free_vars, x_trial)

    eqs = network.residuals()
    f = _scaled_residual_vector(eqs)
    return SolveResult(False, options.max_iter, float(np.linalg.norm(f, ord=2)), "Max iterations reached")
