"""CartPole task environment configuration."""

import math
from dataclasses import dataclass, field
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
  ObservationGroupCfg as ObsGroup,
  ObservationTermCfg as ObsTerm,
  RewardTermCfg as RewardTerm,
  TerminationTermCfg as DoneTerm,
  EventTermCfg as EventTerm,
  term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_cartpole.robot.cartpole_constants import CARTPOLE_ROBOT_CFG
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform
from mjlab.envs import mdp

SCENE_CFG = SceneCfg(
  num_envs=512,
  extent=1.0,
  entities={"robot": CARTPOLE_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="pole",
  distance=3.0,
  elevation=10.0,
  azimuth=90.0,
)


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=20.0,
    use_default_offset=False,
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    cart_pos: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qpos[:, 0:1])
    angle: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qpos[:, 1:2])
    cart_vel: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qvel[:, 0:1])
    ang_vel: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qvel[:, 1:2])

  @dataclass
  class CriticCfg(PolicyCfg):
    pass

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)


def compute_upright_reward(env):
  return env.sim.data.qpos[:, 1].cos()


def compute_center_reward(env, std=0.3):
  return torch.exp(-(env.sim.data.qpos[:, 0] ** 2) / (2 * std**2))


def compute_effort_penalty(env):
  return -(env.sim.data.ctrl[:, 0] ** 2)


@dataclass
class RewardCfg:
  upright: RewardTerm = term(RewardTerm, func=compute_upright_reward, weight=5.0)
  center: RewardTerm = term(RewardTerm, func=compute_center_reward, weight=1.0)
  effort: RewardTerm = term(RewardTerm, func=compute_effort_penalty, weight=1e-2)


def random_push_cart(env, env_ids, force_range=(-5, 5)):
  n = len(env_ids)
  random_forces = (
    torch.rand(n, device=env.device) * (force_range[1] - force_range[0])
    + force_range[0]
  )
  env.sim.data.qfrc_applied[env_ids, 0] = random_forces


def reset_pole(env, env_ids):
  # Cart position is in -2, 2
  env.sim.data.qpos[env_ids, 0] = sample_uniform(-2, 2, len(env_ids), device=env.device)
  # Pole angle is in -0.2, 0.2 radians
  env.sim.data.qpos[env_ids, 1] = sample_uniform(
    -0.1, 0.1, len(env_ids), device=env.device
  )
  # Reset velocities
  env.sim.data.qvel[env_ids, :] = 0.0


@dataclass
class EventCfg:
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=reset_pole,
    mode="reset",
  )
  base_com: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),  # Override in robot cfg.
      "operation": "abs",
      "field": "body_mass",
      "ranges": (0.8, 1.2),
    },
  )


@dataclass
class EventCfgWithPushes(EventCfg):
  random_push: EventTerm = term(
    EventTerm,
    func=random_push_cart,
    mode="interval",
    interval_range_s=(1.0, 2.0),
    params={"force_range": (-20.0, 20.0)},
  )


def check_pole_tipped(env):
  return env.sim.data.qpos[:, 1].abs() > math.radians(30)


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  tipped: DoneTerm = term(DoneTerm, func=check_pole_tipped, time_out=False)


SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.02,
    iterations=1,
  ),
)


@dataclass
class CartPoleEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfgWithPushes = field(default_factory=EventCfgWithPushes)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 1
  episode_length_s: float = 10.0


@dataclass
class CartPoleEnvCfg_PLAY(CartPoleEnvCfg):
  events: EventCfg = field(default_factory=EventCfg)
#   episode_length_s: float = 1e9


@dataclass
class CartPoleRlCfg(RslRlOnPolicyRunnerCfg):
  max_iterations: int = 1_000
