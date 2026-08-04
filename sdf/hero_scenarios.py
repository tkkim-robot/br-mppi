from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import yaml

from sdf.geometry import CircleObstacle, ObstacleField


HERO_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hero_scenarios.yaml"


@dataclass(frozen=True)
class HeroScenario:
    name: str
    robot_name: str
    description: str
    obstacle_field: ObstacleField
    start_state: jnp.ndarray | None
    goal: jnp.ndarray | None


def load_hero_scenarios(path: Path = HERO_CONFIG_PATH) -> dict[str, HeroScenario]:
    """Load all hero scenarios from the YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    scenarios = {}
    for name, config in data.items():
        obstacles = [
            CircleObstacle(center=tuple(obs["center"]), radius=obs["radius"])
            for obs in config["obstacles"]
        ]
        
        start_state = (
            jnp.array(config["start_state"], dtype=float)
            if "start_state" in config
            else None
        )
        goal = (
            jnp.array(config["goal"], dtype=float)
            if "goal" in config
            else None
        )

        scenarios[name] = HeroScenario(
            name=name,
            robot_name=config["robot"],
            description=config.get("description", ""),
            obstacle_field=ObstacleField(obstacles=tuple(obstacles)),
            start_state=start_state,
            goal=goal,
        )
    return scenarios


def get_hero_scenario(name: str, path: Path = HERO_CONFIG_PATH) -> HeroScenario:
    """Load a single hero scenario by name."""
    scenarios = load_hero_scenarios(path)
    if name not in scenarios:
        choices = ", ".join(sorted(scenarios.keys()))
        raise KeyError(f"Unknown hero scenario {name!r}. Choose one of: {choices}")
    return scenarios[name]
