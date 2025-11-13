from mjlab.tasks.registry import register_mjlab_task
from .cartpole_env_cfg import CARTPOLE_ENV_CFG, CARTPOLE_ENV_CFG_PLAY, CARTPOLE_RL_CFG

register_mjlab_task(
    task_id="Mjlab-Cartpole",
    env_cfg=CARTPOLE_ENV_CFG,
    rl_cfg=CARTPOLE_RL_CFG,
)

register_mjlab_task(
    task_id="Mjlab-Cartpole-Play",
    env_cfg=CARTPOLE_ENV_CFG_PLAY,
    rl_cfg=CARTPOLE_RL_CFG,
)
