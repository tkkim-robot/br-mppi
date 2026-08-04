from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdf.hero_scenarios import load_hero_scenarios, get_hero_scenario
from examples.hero_benchmark import run_hero_trial


def test_load_scenarios():
    scenarios = load_hero_scenarios()
    assert len(scenarios) > 0
    assert "narrow_passage" in scenarios
    assert "slalom" in scenarios
    
    scenario = get_hero_scenario("narrow_passage")
    assert scenario.robot_name == "unicycle"
    assert len(scenario.obstacle_field.obstacles) == 2


def test_hero_trial_smoke():
    # Run a very short trial to ensure no crashes
    result, data = run_hero_trial(
        scenario_name="narrow_passage",
        algo="brmppi",
        nsdf=False,
        max_steps=5,
        seed=42
    )
    
    assert result.steps == 5
    assert len(data["trajectory"]) == 6
    assert data["goal"] is not None
    assert data["scenario"].name == "narrow_passage"
