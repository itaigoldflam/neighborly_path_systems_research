"""Shared Gurobi environment/model boilerplate and solve-result helpers.

Consolidates two idioms used interchangeably across the repo (a bare gp.Model with
OutputFlag=0, vs. an explicit quiet gp.Env) into one canonical, silent-by-default
path, plus a single result shape for "did it solve, and if not why" questions that
call sites previously re-implemented ad hoc as bools, (bool, list, list) tuples, or
dicts with inconsistent keys.
"""
from dataclasses import dataclass
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB


def quiet_env() -> gp.Env:
    """Create and start a Gurobi Env with console output suppressed.

    Caller owns the returned Env (close it, or let it be garbage collected).
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def quiet_model(name: str = "", env: Optional[gp.Env] = None) -> gp.Model:
    """Create a silent Gurobi Model. Creates and owns a quiet Env if none is given."""
    if env is None:
        env = quiet_env()
    return gp.Model(name, env=env)


@dataclass
class SolveResult:
    """Canonical outcome of optimize() + status check."""
    status: int
    feasible: bool
    iis: Optional[List[str]] = None


def extract_iis(model: gp.Model) -> List[str]:
    """Compute and return the constraint names in the model's Irreducible Inconsistent
    Subsystem. The model must already be infeasible (Gurobi requirement)."""
    model.computeIIS()
    return [c.ConstrName for c in model.getConstrs() if c.IISConstr]


def optimize_and_status(model: gp.Model, *, compute_iis: bool = False) -> SolveResult:
    """Run model.optimize() and return a canonical SolveResult.

    If compute_iis is True and the model turns out infeasible, the IIS is computed
    automatically; otherwise `iis` is left as None so callers that don't need it
    don't pay for it.
    """
    model.optimize()
    feasible = model.Status == GRB.OPTIMAL
    iis = extract_iis(model) if (compute_iis and model.Status == GRB.INFEASIBLE) else None
    return SolveResult(status=model.Status, feasible=feasible, iis=iis)
