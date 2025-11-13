"""CartPole task environment configuration."""

import math
from copy import deepcopy
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_cartpole.robot.cartpole_constants import CARTPOLE_ROBOT_CFG
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform
from mjlab.envs import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.retval import retval


SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="plane",
  ),
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

SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.02,
    iterations=1,
  ),
)


@retval
def CARTPOLE_ENV_CFG() -> ManagerBasedRlEnvCfg:
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.JointEffortActionCfg(
      asset_name="robot",
      actuator_names=("slide",),
      scale=20.0,
    )
  }

  policy_terms = {
    "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1]),
    "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2]),
    "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1]),
    "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2]),
  }

  critic_terms = {**policy_terms}

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  def compute_upright_reward(env):
    return env.sim.data.qpos[:, 1].cos()

  def compute_center_reward(env, std=0.3):
    return torch.exp(-(env.sim.data.qpos[:, 0] ** 2) / (2 * std**2))

  def compute_effort_penalty(env):
    return -((env.sim.data.ctrl[:, 0] / 20) ** 2)

  rewards = {
    "upright": RewardTermCfg(func=compute_upright_reward, weight=5.0),
    "center": RewardTermCfg(func=compute_center_reward, weight=1.0),
    "effort": RewardTermCfg(func=compute_effort_penalty, weight=1e-2),
  }

  def check_pole_tipped(env):
    return env.sim.data.qpos[:, 1].abs() > math.radians(30)

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "tipped": TerminationTermCfg(func=check_pole_tipped, time_out=False),
  }

  def random_push_cart(env, env_ids, force_range=(-5, 5)):
    n = len(env_ids)
    random_forces = (
      torch.rand(n, device=env.device) * (force_range[1] - force_range[0])
      + force_range[0]
    )
    env.sim.data.qfrc_applied[env_ids, 0] = random_forces

  def reset_pole(env, env_ids):
    # Cart position is in -2, 2
    env.sim.data.qpos[env_ids, 0] = sample_uniform(
      -2, 2, len(env_ids), device=env.device
    )
    # Pole angle is in -0.2, 0.2 radians
    env.sim.data.qpos[env_ids, 1] = sample_uniform(
      -0.1, 0.1, len(env_ids), device=env.device
    )
    # Reset velocities
    env.sim.data.qvel[env_ids, :] = 0.0

  events = {
    "reset_robot_joints": EventTermCfg(func=reset_pole, mode="reset"),
    "base_com": EventTermCfg(
      func=mdp.randomize_field,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=[".*"]
        ),  # Override in robot cfg.
        "operation": "abs",
        "field": "body_mass",
        "ranges": (0.8, 1.2),
      },
    ),
    "random_push": EventTermCfg(
      func=random_push_cart,
      mode="interval",
      interval_range_s=(1.0, 2.0),
      params={"force_range": (-20.0, 20.0)},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SCENE_CFG,
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminations=terminations,
    events=events,
    sim=SIM_CFG,
    viewer=VIEWER_CONFIG,
    decimation=1,
    episode_length_s=20.0,
  )


@retval
def CARTPOLE_ENV_CFG_PLAY() -> ManagerBasedRlEnvCfg:
  cfg = deepcopy(CARTPOLE_ENV_CFG)

  # No random push while in play env
  del cfg.events["random_push"]

  return cfg


CARTPOLE_RL_CFG = RslRlOnPolicyRunnerCfg(
  max_iterations=500, wandb_project="mjlab_cartpole"
)
