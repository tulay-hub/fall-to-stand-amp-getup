# Lens110「倒地自动起身」实机部署包 v2 (2026-09-08)

训练来源: `lens110-amp-mjlab/AMP_mjlab/logs/rsl_rl/getup_v5/2026-09-08_15-43-38_v5f`，
权重 `model_72500.pt` 及由其导出的 ONNX（两者同一时刻落盘，已核对 mtime）。
框架 AMP_mjlab（mjlab + AMP 版 rsl_rl），物理 500 Hz / 策略 100 Hz，PD 取自 Lens110
实机配置 `control.walk` 档（**踝刚度本包起为 30.0，不是 v1 的 5.0**）。

> 部署前必须先读 `docs/KNOWN_ISSUES.md` 第 1 条：策略与 PD 增益必须配对，
> 用 v1 的 kp=5 播这份 onnx 会横移出十几米。

## 实测能力（离线 MuJoCo，训练同款 MJCF + 官方平脚站姿出生）

| 测项 | 结果 |
|---|---|
| 从倒地帧起身成功率（107 个采样帧，8 s，无推力） | **100%**（极低≤0.30 m 35/35、低 0.30–0.44 40/40、中 0.44–0.60 32/32） |
| 起身用时中位（从 ≤0.30 m 出生） | **0.79 s** |
| 站立期骨盆高度 | 0.638 ~ 0.660 m，均值 0.647（平脚贴地真值 0.656756） |
| 脚掌倾角（站立 + 每 6 s 轻推 30 s） | 左 **0.84°** / 右 **1.33°**，toe-heel −1.7 / −3.5 mm |
| 左右脚承重 | 112.2 N / 112.8 N（合计 225 N = 体重，几乎对半分） |
| 被推后水平位移 | 最远 15.6 cm，末态回到 (−7.3, +2.6) cm |
| 蹬地腾空 | 未出现（最大 0.660 m；v5e 曾冲到 0.925 / 2.597 m） |

对照 v1：同一套"4 秒内站起率"测法，v1 是 **3.5%**，且末态瘫在判摔线上；
v1 站立时左踝被主动指令顶到脚掌倾角 31°、压力中心 +96 mm（纯踮脚尖），左右脚承重 88 N vs 137 N。

## 这个策略不需要动作文件

actor 观测 288 维、纯本体状态（见 `docs/OBSERVATION_ORDER.md`），**不含 motion reference、
不含地形扫描**。实机只需要 IMU + 关节编码器 + 上一帧指令，不需要加载 npz、不需要时间轴。
`command` 恒为 `[0,0,0]`——基准行为就是"站着什么都不动"。

## 目录

```
policy/     Lens110-AMP-GetUp_model_72500.onnx    推理入口 (288 -> 21)
            model_72500.pt                         原始 rsl_rl checkpoint（含观测归一化参数）
            params/env.yaml, agent.yaml            训练时完整配置快照（权威依据）
            tensorboard_events.tfevents.0          训练曲线
assets/     init_states/*.npz                      离线验证用出生姿态（不是推理参考）
config/     deploy_config.yaml                     PD/关节顺序/scale/频率/动作公式/SDK 置换
docs/       OBSERVATION_ORDER.md                   288 维逐段说明 + 历史排布
            KNOWN_ISSUES.md                        已知问题与限制，部署前必读
            CHANGES_v1_to_v2.md                    这一版相对 v1 改了什么、为什么
            REAL_ROBOT_PARAMS.json                 逐关节参数表（机读，按训练侧权威值重生成）
replay/     play_lens110_amp_getup.py              MuJoCo 离线验证（可手动摆姿/推倒）
            mjcf/lens110.xml + meshes/*.STL        验证用模型（除 meshdir 外与训练用 XML 逐字节一致）
```

## 推理接口

```
输入  obs      float32 [1, 288]    原始物理量，不要自己归一化
输出  actions  float32 [1, 21]      网络输出，按 MJCF 关节顺序
```

ONNX 图内已带观测归一化（`Sub(obs, obs_normalizer._mean)` + `Div`），外部只要按
`docs/OBSERVATION_ORDER.md` 的顺序喂 288 维原始值。

动作换算：`q_des = default_joint_pos_rad + action_scale ⊙ action`，**不 clip action**。
`default_joint_pos_rad` 已是官方平脚站姿原值（膝 0.36、踝 pitch −0.23804、髋 roll 左 −0.01／右 +0.01、
两肘均 −0.8），所以策略输出零动作时机器人天然就是平脚站。

## 关节顺序（唯一会咬人的地方）

策略向量是 **MJCF 顺序**（与 `lens110_21dof.urdf` / 同名 USD 逐位一致），
实机 SDK 数组顺序不同，置换已写进 `config/deploy_config.yaml → real_robot_mapping.joint_order`：

```
SDK[k] = MJCF[joint_order[k]]
joint_order = [0,1,2,3,4,5,6,7,8,9,10,11, 17,18,19,20, 13,14,15,16, 12]
              └──── 双腿(踝按 upper/lower) ────┘ └─ 左臂4 ─┘ └─ 右臂4 ─┘ torso
```

即：前 12 个位置一致；SDK 里左臂在 12~15、右臂在 16~19、`torso_yaw` 排最后。
本包**没有**置换 onnx 的输入输出顺序（仍是 MJCF），置换放在控制器侧。

## 离线自测

```bash
cd replay
/home/tulay/miniconda3/envs/gmr/bin/python -u play_lens110_amp_getup.py \
  --onnx ../policy/Lens110-AMP-GetUp_model_72500.onnx \
  --init-store ../assets/init_states --init stand --push-every 6
```

按键：空格 暂停/继续 · `r` 重生 · `f` 手动推一把 · `n` 切换出生方式
（default → stand → recovery）· `[` `]` 调速 · `Esc` 退出。
