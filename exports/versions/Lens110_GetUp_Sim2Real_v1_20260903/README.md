# Lens110「倒地自动起身」实机部署包 v1 (2026-09-03)

训练来源: `lens110-amp-mjlab/AMP_mjlab/logs/rsl_rl/getup_60k/2026-09-02_18-28-52`，
权重 `model_54500.pt` 及由其导出的 ONNX。框架为 AMP_mjlab (mjlab + AMP 版 rsl_rl)，
物理 500 Hz、策略 100 Hz，PD 全部取自 Lens110 自己的实机配置。

## 这个策略不需要动作文件

actor 观测是 288 维、纯本体状态（见 `docs/OBSERVATION_ORDER.md`），
**不含 motion reference、不含地形高度扫描**。所以实机侧只需要 IMU + 关节编码器 +
上一帧指令，不需要加载 npz、不需要规划器、不需要时间轴。这是这版方案对部署最友好的地方。

速度指令 `command` 恒为 `[0, 0, 0]`：基准行为就是"站着什么都不动"。

## 目录

```
policy/     Lens110-AMP-GetUp_model_54500.onnx   推理入口 (288 -> 21)
            model_54500.pt                        原始 rsl_rl checkpoint (含观测归一化参数)
            params/env.yaml, agent.yaml           训练时完整配置快照 (权威依据)
            tensorboard_events.tfevents.0         训练曲线
assets/     init_states/*.npz                   离线验证用的出生姿态(不是推理参考)
config/     deploy_config.yaml                    PD/关节顺序/scale/频率/动作公式/映射关系
docs/       OBSERVATION_ORDER.md                  288 维逐段说明 + 历史排布
            KNOWN_ISSUES.md                       复现环境已知缺陷, 部署前必读
            REAL_ROBOT_PARAMS.json                逐关节参数表 (机读)
replay/     play_lens110_amp_getup.py             MuJoCo 离线验证 (含手动摆姿/推倒)
            mjcf/lens110.xml + meshes/*.STL       验证用模型 (MJCF 顺序 21 关节)
```

## 推理接口

```
输入  obs      float32 [1, 288]   原始值, 不要自己归一化
输出  actions  float32 [1, 21]     网络输出, 按 MJCF 关节顺序
```

ONNX 图里已经带上了观测归一化（节点 `Sub(obs, obs_normalizer._mean)` + `Div`），
所以外部只要按 `docs/OBSERVATION_ORDER.md` 的顺序把 288 维**原始物理量**喂进去。

每个策略步（10 ms）：

```python
q_des = default_joint_pos + action_scale * action      # 不 clip action
# 目标角不做限位钳制; 500 Hz 下执行 PD: tau = kp*(q_des-q) + kd*(0-qd), 限幅到 effort_limit
```

`default_joint_pos`、`action_scale`、`kp/kd`、`effort_limit` 全部按
`config/deploy_config.yaml` 里的**逐关节数组**（顺序是 MJCF 顺序，见下）。

## 关节顺序与实机映射（最容易踩的地方）

网络与 `deploy_config.yaml` 用的都是 **MJCF 顺序**：

```
左腿6(hip_p,hip_r,hip_y,knee,ankle_pitch,ankle_roll),
右腿6(同上), torso_yaw, 右臂4, 左臂4
```

而实机 `robot_humanoid_lens110_config.yaml` 是：左腿6 + 右腿6 + **左臂4 + 右臂4 + torso_yaw(第21位)**，
并带 `joint_direction` 的 ±1 符号。两者顺序不同，必须按
`config/deploy_config.yaml: real_robot_mapping` 做置换 + 符号 + 踝名映射
（`ankle_pitch ↔ ankle_upper`、`ankle_roll ↔ ankle_lower`，同名同侧；
真正的双电机耦合请复用 `legged_lab_lbot/scripts/sim2sim_ul/ul_full.py` 里已有的实现）。

## 离线验证（不需要实机）

```bash
conda activate gmr      # 有 mujoco 3.11 + onnxruntime 1.23
cd Lens110_GetUp_Sim2Real_v1_20260903/replay
python play_lens110_amp_getup.py                          # 理想站姿出生
python play_lens110_amp_getup.py --init stand             # 训练里的站立出生帧
python play_lens110_amp_getup.py --init recovery --lowest # 从倒地最低帧出生, 看会不会起身
python play_lens110_amp_getup.py --init default --push-every 2.0   # 每 2 秒随机推
```

窗口按键：`空格` 暂停/继续、`r` 重新出生、`f` 施加一次随机推力、`n` 切换出生方式、
`[` `]` 调慢/调快、`Esc` 退出。每 50 个策略步打印骨盆高度、`>0.62 m` 时间占比、关节速度峰值。

已跑过的结果（22 秒、8 倍速，脚本的观测拼装与 mjlab 训练环境逐项对照过）：
理想站姿出生 `69%` 时间在 0.62 m 以上但在 0.37~0.71 m 之间起伏；
训练站立出生帧只有 `14.7%`；从倒地最低帧出生是 `0.0%`，末态都收敛到约 `0.350 m`。
详见 `docs/KNOWN_ISSUES.md`。

## 先读 KNOWN_ISSUES.md

这版权重**没有真正学会从倒地起身**（实测从 demo 倒地帧出生，4 秒内站起率 3.5%），
它擅长的是"站着不被推倒"。上线前请先看 `docs/KNOWN_ISSUES.md`。
