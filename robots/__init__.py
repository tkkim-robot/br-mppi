from robots.base_robot import RobotModel
from robots.dynamic_unicycle import DynamicUnicycleRobot
from robots.mobile_arm import MobileArmRobot
from robots.planar_quadrotor import PlanarQuadrotorRobot
from robots.single_integrator import SingleIntegratorRobot
from robots.unicycle import UnicycleRobot


ROBOT_REGISTRY = {
    "single_integrator": SingleIntegratorRobot,
    "unicycle": UnicycleRobot,
    "dynamic_unicycle": DynamicUnicycleRobot,
    "planar_quadrotor": PlanarQuadrotorRobot,
    "mobile_arm": MobileArmRobot,
}


def create_robot(name: str) -> RobotModel:
    try:
        return ROBOT_REGISTRY[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(ROBOT_REGISTRY))
        raise ValueError(f"Unknown robot '{name}'. Choose one of: {choices}") from exc


__all__ = [
    "DynamicUnicycleRobot",
    "MobileArmRobot",
    "PlanarQuadrotorRobot",
    "ROBOT_REGISTRY",
    "RobotModel",
    "SingleIntegratorRobot",
    "UnicycleRobot",
    "create_robot",
]
