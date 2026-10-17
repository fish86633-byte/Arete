"""Portable copies of the paper parameters; no historical filesystem access."""

from importlib.resources import files
import json
import math


def paper_config(rho=0.4):
    """Return independent paper parameters, with R_total = rho * C_ref.

    This function does not generate a constellation or traffic samples. The
    original registered configuration and the two derived fields are documented
    in provenance/source_manifest.json.
    """
    rho = float(rho)
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    config = json.loads(files("arete").joinpath("data/paper_s24_rho04.json").read_text())
    config["load_ratio"] = rho
    config["R_total"] = rho * config["C_ref"]
    return config


def physical_constants():
    """Return the numerical constants used by the frozen physical model."""
    return json.loads(files("arete").joinpath("data/constants.json").read_text())
