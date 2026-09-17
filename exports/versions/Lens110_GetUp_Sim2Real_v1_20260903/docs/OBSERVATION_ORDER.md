# 288 维观测布局 (必须严格一致)

排布方式 **time-major**：先按时间步分块，每块内 72 维。历史长度 4，策略频率 100 Hz，
所以 4 块覆盖最近 40 ms，其中**最后一块是当前帧**。

```
obs[  0: 72]  = t-3 帧 (最旧)
obs[ 72:144]  = t-2 帧
obs[144:216]  = t-1 帧
obs[216:288]  = t   帧 (当前, 推理时刻)
```

每一块 72 维内部顺序（就是 `config/deploy_config.yaml: observation.terms`）：

| 偏移 | 项 | 维度 | 内容 | 单位 / 坐标系 |
|---|---|---|---|---|
| 0 | `base_ang_vel` | 3 | 骨盆（IMU site）角速度，取自 `imu_ang_vel` 陀螺 | rad/s，世界系 |
| 3 | `projected_gravity` | 3 | 重力向量旋到骨盆系：`R_pelvisᵀ · [0,0,-1]` | 无量纲，站立时 ≈ `[0,0,-1]` |
| 6 | `command` | 3 | 速度指令 `[vx, vy, yaw_rate]`，**恒为 0** | m/s, rad/s |
| 9 | `joint_pos_rel` | 21 | `q - default_joint_pos`（不是绝对角） | rad |
| 30 | `joint_vel_rel` | 21 | 关节速度 | rad/s |
| 51 | `prev_actions` | 21 | 上一步网络原始输出（未乘 scale、未加 default） | 无量纲 |

## 关节顺序

第 9~72 项的 21 个分量都是 **MJCF 顺序**：

```
0  left_hip_pitch     1  left_hip_roll      2  left_hip_yaw
3  left_knee          4  left_ankle_pitch   5  left_ankle_roll
6  right_hip_pitch    7  right_hip_roll     8  right_hip_yaw
9  right_knee        10  right_ankle_pitch  11  right_ankle_roll
12 torso_yaw         13  right_shoulder_pitch  14 right_shoulder_roll
15 right_shoulder_yaw 16 right_elbow        17  left_shoulder_pitch
18 left_shoulder_roll  19 left_shoulder_yaw  20  left_elbow
```

注意与实机 `robot_humanoid_lens110_config.yaml` 不同：实机是左臂在前、且 `torso_yaw` 排在第 21 位。

## 归一化与噪声

* 观测归一化**已经在 ONNX 图内**（`Sub(obs, obs_normalizer._mean)` 然后 `Div`，
  `_mean` 长度 288、`var` 对应 `add_3`）。外部喂原始物理量即可，**不要再做归一化**。
* 训练时 actor 观测带仿真噪声：`base_ang_vel ±0.2`、`projected_gravity ±0.05`、
  `joint_pos ±0.01`、`joint_vel ±0.5`；`command`、`prev_actions` 无噪声。
  这些只是训练鲁棒化，实机不要人为加噪。
* 历史缓冲初始化：episode 开始时 4 帧都用**当前帧观测复制**填充。
* 动作裁剪：训练里 `clip_actions=None`，即网络输出不截断；只有 PD 力矩受 `effort_limit` 限制。

## 输出接口

```
actions: float32 [21]   按同一 MJCF 顺序
q_des[i] = default_joint_pos[i] + action_scale[i] * actions[i]
```

`action_scale`（每关节，与 MJCF 顺序一致，也见 `deploy_config.yaml`）：

```
髋×3、膝: 0.5      踝 pitch/roll: 0.25
肩×3、肘: 0.45     torso_yaw: 0.2
```

（等于 `0.25 × effort_limit / kp`，踝单独取 0.25 是因为 walk 档踝 kp 只有 5，
按公式会得到 1.8 rad 的缩放导致指令长期饱和。）
