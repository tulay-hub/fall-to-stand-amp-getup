"""Lens110 AMP 任务注册: 平地, 基准动作=站立不动, 目标=倒地自动起身。"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import lens110_amp_getup_env_cfg
from .rl_cfg import lens110_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Lens110-AMP-GetUp",
  env_cfg=lens110_amp_getup_env_cfg(),
  play_env_cfg=lens110_amp_getup_env_cfg(play=True),
  rl_cfg=lens110_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
