"""Bind the paper's M1-E wrapper to its U8 constructor."""

from functools import partial
import math
from threading import RLock

import numpy as np

from .config import paper_config, physical_constants
from .model import Channel
from ._frozen.confirmation_010 import marginal_wrapper
from ._frozen.pcqr.support import Meter
from ._frozen.speed_validation_018.construct import construct_proposal

# The frozen implementation binds two constructors temporarily at module scope.
# Serialize public calls in one process, restoring both bindings on exceptions.
# Separate processes are the supported route for parallel experiments.
_PLANNER_LOCK = RLock()


def plan(state, arrivals, satellite_positions, cells, satellite_ids, *,
         config=None, constants=None, cycle=0):
    """Plan one complete horizon with the paper's soft-budget semantics.

    Positions are ECEF metres: satellite_positions[H,S,3], cells[M,3].
    arrivals[H,M,2] contains integer RT/NRT packet counts. `state` precedes
    arrivals at absolute slot cycle*H; NRT batches retain their birth slots.

    Returns the frozen result dictionary plus `algorithm` and `channel`.
    `plan` is a boolean [H,S,M] assignment array. Inspect `run_status`, `stop`,
    `reliable_plan_source` and `fallback_used` before using a result. Independent
    verification is available separately through `arete.verify_plan`.

    The planning meter starts before Channel construction and includes the
    frozen preparation, paid TPG reference, construction, replay and output.
    Package imports, caller-side input loading and independent audit are outside
    that meter. Budget checks are cooperative: atomic work may overrun.
    """
    c = paper_config() if config is None else dict(config)
    const = physical_constants() if constants is None else dict(constants)
    if isinstance(cycle, bool) or int(cycle) != cycle or cycle < 0:
        raise ValueError("cycle must be a nonnegative integer")
    if not math.isfinite(float(c["planning_budget_s"])) or c["planning_budget_s"] < 0:
        raise ValueError("planning_budget_s must be finite and nonnegative")
    positions = np.asarray(satellite_positions)
    ground = np.asarray(cells)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] != c["H"]:
        raise ValueError("satellite_positions must have shape [configured H,S,3]")
    if ground.ndim != 2 or ground.shape[1] != 3:
        raise ValueError("cells must have shape [M,3]")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(ground)):
        raise ValueError("positions must be finite")
    with _PLANNER_LOCK:
        meter = Meter(c["planning_budget_s"])
        try:
            with meter.phase("geometry_visibility"):
                channel = Channel(positions, ground, satellite_ids, c, const)
        except BaseException:
            meter.finish()
            raise
        previous = marginal_wrapper.construct_proposal
        marginal_wrapper.construct_proposal = partial(construct_proposal, implementation="UB")
        try:
            result = marginal_wrapper.plan(state, arrivals, channel,
                                          cycle=int(cycle), meter=meter, variant="M1-E")
        finally:
            marginal_wrapper.construct_proposal = previous
            if meter.ended is None:
                meter.finish()
        result["algorithm"] = "Arete"
        result["channel"] = channel
        return result
