"""R2 v0.1.1 queue-snapshot preparation and early silent safety registration.

These helpers do not choose service edges, alter the queue contract, or compute
an objective for an un-replayed fallback plan.
"""
from __future__ import annotations

import numpy as np

from .engine import State


def register_silent(channel, supplied=None):
    """Register a legal all-silent plan before any expensive state rebuilding."""
    shape = (channel.H, channel.S, channel.M)
    if supplied is None:
        plan = np.zeros(shape, dtype=bool)
        registration = {"registered_before_preparation": True, "provided_by_driver": False}
    else:
        plan = supplied.get("plan")
        if not isinstance(plan, np.ndarray) or plan.dtype != np.bool_ or plan.shape != shape:
            raise ValueError("Registered silent fallback must be a bool H-by-S-by-M array")
        if np.any(plan):
            raise ValueError("Registered silent fallback must contain no scheduled beam")
        registration = {"registered_before_preparation": True, "provided_by_driver": True,
                        "driver_registration": {key: value for key, value in supplied.items() if key != "plan"}}
    return plan, registration


def prepare_inputs(Q0, A, channel, cycle, meter, ledger):
    """Consume the decoded protocol payload with one authoritative construction.

    The FIFO/RT constructor owns its own preparation subpart instrumentation.
    The decoded message is not mutated because it remains protocol evidence.
    """
    telemetry = meter.preparation
    with meter.phase("traffic_preparation"):
        # Codec timing remains under Ledger's exclusive codec_protocol phase.
        # PreparationMetrics reconciliation excludes that phase from its seven
        # preparation subparts; no part() context encloses this emit call.
        obs = ledger.emit("QUEUE_SCENE_SNAPSHOT", "gateway", ["gateway"],
                          {"cycle": cycle, "rt": Q0.rt,
                           "nrt_batches": Q0.nrt_batches, "arrivals": A})
        with telemetry.part("message_postprocessing"):
            if obs["cycle"] != cycle:
                raise ValueError("queue snapshot cycle mismatch")
            rt = obs["rt"]
            batches = obs["nrt_batches"]
            arrivals = obs["arrivals"]
        meter.check()
        state = State.from_decoded(rt, batches, telemetry=telemetry)
        meter.check()
        with telemetry.part("array_conversion"):
            scene = np.asarray(arrivals, dtype=np.int64)
            telemetry.copied(scene.nbytes, "decoded_scene_array")
        with telemetry.part("item_validation"):
            if scene.shape != (channel.H, channel.M, 2):
                raise ValueError("decoded scene must have shape H by M by 2")
            if np.any(scene < 0):
                raise ValueError("decoded scene contains negative packet counts")
        with telemetry.part("message_postprocessing"):
            # This only drops local references. Ledger retains its entire
            # decoded payload and byte accounting for independent inspection.
            del obs, rt, batches, arrivals
    return state, scene
