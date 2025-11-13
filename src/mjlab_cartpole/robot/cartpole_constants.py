from pathlib import Path
import mujoco

import os
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlMotorActuatorCfg

CARTPOLE_XML: Path = Path(os.path.dirname(__file__)) / "xmls" / "cartpole.xml"
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


CARTPOLE_ROBOT_CFG = EntityCfg(
  spec_fn=get_spec,
  articulation=EntityArticulationInfoCfg(
    actuators=(XmlMotorActuatorCfg(joint_names_expr=("slide",)),)
  ),
)

if __name__ == "__main__":
  import mujoco.viewer as viewer

  robot = Entity(CARTPOLE_ROBOT_CFG)
  viewer.launch(robot.spec.compile())
