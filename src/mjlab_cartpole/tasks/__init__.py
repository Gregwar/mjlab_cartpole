from mjlab.tasks.registry import register_mjlab_task
from .cartpole_env_cfg import cartpole_env_cfg, cartpole_rl_cfg

register_mjlab_task(
  task_id="Mjlab-Cartpole",
  env_cfg=cartpole_env_cfg(),
  play_env_cfg=cartpole_env_cfg(play=True),
  rl_cfg=cartpole_rl_cfg,
)
