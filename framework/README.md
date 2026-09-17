# 跌倒起身 AMP 框架入口

实体源码在 `amp_mjlab/AMP_mjlab`。它是独立的 mjlab/AMP 工程，不与共享 Isaac Lab DeepMimic/DWAQ 源码
合并。`data/training/amp_lens110_motions` 只读指向框架内 Lens110 Stand/Recovery motion 目录；导出包统一
放在项目自己的 `exports/versions`。
