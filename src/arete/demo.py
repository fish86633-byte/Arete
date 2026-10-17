"""Small deterministic synthetic scene, independent of the paper input banks."""

import numpy as np

from .config import paper_config, physical_constants
from .model import State, verify_plan
from .planner import plan


def ecef(latitude, longitude, radius):
    latitude, longitude = np.deg2rad([latitude, longitude])
    return radius * np.array([np.cos(latitude) * np.cos(longitude),
                              np.cos(latitude) * np.sin(longitude), np.sin(latitude)])


def demo_inputs():
    """Six regional cells and two stationary satellites for an integration test.

    H, TTL, power, antennas and objective normalizers retain paper values. The
    hand-specified geometry and packet counts are demonstration inputs only.
    """
    config, constants = paper_config(), physical_constants()
    radius = constants["earth_radius_m"]
    cells = np.array([ecef(lat, lon, radius) for lat, lon in
                      [(29.8, 109.8), (29.8, 110.0), (29.8, 110.2),
                       (30.2, 109.8), (30.2, 110.0), (30.2, 110.2)]])
    satellites = np.array([ecef(30.0, lon, radius + config["altitude_m"])
                           for lon in (109.5, 110.5)])
    positions = np.repeat(satellites[None], config["H"], axis=0)
    rt = np.zeros((len(cells), config["ttl"]), dtype=np.int64)
    rt[:, 10], rt[:, 40], rt[:, 59] = 300, 400, 200
    state = State(rt, [[(-10, 1800), (-1, 1200)] for _ in cells])
    arrivals = np.zeros((config["H"], len(cells), 2), dtype=np.int64)
    arrivals[:, :, 0], arrivals[:, :, 1] = 40, 100
    return state, arrivals, positions, cells, np.array([101, 202]), config, constants


def run_demo():
    state, arrivals, positions, cells, ids, config, constants = demo_inputs()
    result = plan(state, arrivals, positions, cells, ids, config=config, constants=constants)
    if result["run_status"] != "OK" or not result["output_published"]:
        raise RuntimeError(f"Arete failed: {result.get('error', result['stop'])}")
    audit = verify_plan(result["plan"], state, arrivals, result["channel"],
                        expected=result["trace"])
    return {"scenario": "synthetic demonstration; not a paper result",
            "algorithm": result["algorithm"], "horizon_slots": config["H"],
            "satellites": len(ids), "cells": len(cells),
            "stop": result["stop"], "accepted": result["accepted"],
            "reliable_plan_source": result["reliable_plan_source"],
            "fallback_used": result["fallback_used"],
            "planning_wall_s": result["costs"]["total_s"],
            "audit": audit.summary()}
