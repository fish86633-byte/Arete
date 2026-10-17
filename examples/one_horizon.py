"""Run from an installed checkout: python examples/one_horizon.py."""

import json
from arete.demo import run_demo

print(json.dumps(run_demo(), indent=2))
