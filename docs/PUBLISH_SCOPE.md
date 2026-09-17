# 发布范围 / Publish Scope

## 中文

本仓库只保留 MuJoCo/AMP GetUp 可复现所需内容和最新模型：

- `exports/versions/Lens110_GetUp_Sim2Real_v2_20260908/`：最新导出包，含 MuJoCo XML、网格、deploy config、`model_72500.pt` 和对应 ONNX；
- `framework/amp_mjlab/AMP_mjlab/logs/rsl_rl/getup_v5/2026-09-08_15-43-38_v5f/model_77999.pt`：最新训练 checkpoint，及其 `params/`；
- AMP/MuJoCo 源码、21-DoF Stand/Recovery 动作、必要配置和回放说明。

旧训练 run、中间 checkpoint、TensorBoard 日志、旧导出包和重复压缩包没有进入本次 Public commit；原始工作区仍保留它们。`.pt` checkpoint 是完整训练状态，导出策略在 `exports/` 中单独提供。

## English

This repository keeps only the files required for reproducible AMP GetUp/MuJoCo use and the newest models:

- `exports/versions/Lens110_GetUp_Sim2Real_v2_20260908/`: newest export with MuJoCo XML, meshes, deploy config, `model_72500.pt`, and its ONNX policy;
- `framework/amp_mjlab/AMP_mjlab/logs/rsl_rl/getup_v5/2026-09-08_15-43-38_v5f/model_77999.pt`: newest training checkpoint with its `params/`;
- AMP/MuJoCo source, 21-DoF Stand/Recovery motions, required configuration, and replay documentation.

Old runs, intermediate checkpoints, TensorBoard logs, old exports, and duplicate archives are not part of this Public commit; the original workspace still retains them. A `.pt` checkpoint contains full training state, while the exported policy is provided separately under `exports/`.
