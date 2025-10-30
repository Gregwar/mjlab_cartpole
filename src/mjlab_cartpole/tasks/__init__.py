import gymnasium as gym

gym.register(
  id="Mjlab-Cartpole",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartPoleEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartPoleRlCfg",
  },
)

gym.register(
  id="Mjlab-Cartpole-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartPoleEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartPoleRlCfg",
  },
)