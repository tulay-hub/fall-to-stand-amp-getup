# 复现说明: 代码已回退到 `getup_60k/2026-09-02_18-28-52` 训练时的状态

日期: 2026-09-03。依据是该 run 自己落盘的配置快照
`logs/rsl_rl/getup_60k/2026-09-02_18-28-52/params/env.yaml` 与 `agent.yaml`
(不是凭记忆改代码)。回退后用脚本把"当前配置"与快照逐字段比对, 30 项全部一致:

```
PASS  sim timestep (0.002) / decimation (5) / episode_length_s (20)
PASS  bad_base_height (0.35) / bad_orientation (70°)
PASS  push interval (1.0,3.0)s / x±0.6 y±0.4 z±0.2 roll·pitch±0.4 yaw±0.6
PASS  delay_reset_env_ratio 0.5 / max_delay_steps 500 / motion_dir Stand / recovery_dir Recovery
PASS  track_root_height = src.tasks.amp_loco.mdp.rewards.track_root_height
      weight 1.0, std 0.25, mask_delay true, delay_env_rew_ratio 3.5
PASS  twist ranges 全 0 (基准动作=站着不动)
PASS  viewer enable_shadows/enable_reflections = True, body_name torso_yaw_link
PASS  terrain textures/materials = mjlab 默认 checker + reflectance 0.2
PASS  actor 观测项与 history_length=4 / action scale / curriculum / events 列表
PASS  amp_task_reward_lerp 0.85 / amp_reward_coef 0.1 / [1024,512,256] / 21 维 min_std
PASS  amp_body_names(11) / amp_anchor_name torso_yaw_link
num_envs 快照里是 2048 而 cfg 默认 1 —— 因为快照含当时的命令行覆盖, 复现时加
`--env.scene.num-envs 2048` 即可。
```

## 具体回退了什么

| 文件 | 现在(=当时) | 被撤销的改动 |
|---|---|---|
| `AMP_mjlab/src/tasks/amp_loco/config/lens110/env_cfgs.py` | push (1.0,3.0) ±0.6/0.4/0.2/0.4/0.6; delay 0.5; 站高用原 mdp 函数 std 0.25 ×3.5; 无 viewer/材质覆盖 | 我后来的: push 降到 (2.5,5.0) 与更柔; std 0.14 + weight 5.0 + 去掉 mask 条件; 关阴影/反射/换地面纹理; `force_recovery_reset` 事件注册 |
| `.../lens110/rl_cfg.py` | `amp_task_reward_lerp=0.85` | 我改的 0.60 |
| `.../lens110/rewards.py` `events.py` | 已移出项目(当时不存在这两个文件) | 我新写的 `track_root_height_grouped` 与 `force_recovery_reset` |
| `AMP_mjlab/src/assets/robots/lens110/xmls/lens110.xml` | 恢复自带 `floor` geom | 我为修地面闪烁删掉的 floor |
| `AMP_mjlab/src/assets/motions/lens110/Stand/` | 恢复 8 段 2172 帧(从 dejerk 站立段切) | 我合成的 800 帧"安静站立"(留档见下) |
| `Recovery/` 8 段 2888 帧 | 未变 | — |

## 已知问题: 这份复现环境带着两个当时就存在的缺陷

1. **两层共面地面**: MJCF 自带 `floor`(plane, 100×100, conaffinity=15) 与 mjlab 平面地形生成的
   `terrain` PLANE 同在 z=0。后果是渲染层 z-fighting(地面闪烁)以及机器人同时与两个平面算接触。
   修法是删 XML 里的 `floor`(只留 mjlab 的 terrain), 但那就不是 18-28-52 的环境了, 所以这里保持原样。
2. **踝的 visual mesh 越界**: `left/right_ankle_roll_link.STL` 局部 z 范围 -92.4~+103.6mm、x 只有 56mm,
   网格里含一截小腿。好在踝 mesh 在本模型里 `contype=0 conaffinity=0` 不参与碰撞(脚底用的是
   `*_ankle_roll/pitch_collision` 两个 box, 盒底 -32.5mm), 所以只影响观感不影响物理。
   站姿下"脚贴地"的骨盆高度实测为 0.664 m, 与 `STAND_KEYFRAME` 一致。

## 我 2026-09-03 那几处修复的留档位置(不在本包内)

```
/tmp/lens110_fixed_20260903/   改过的 env_cfgs.py / rl_cfg.py / rewards.py / events.py / 无 floor 的 lens110.xml
/tmp/lens110_quiet_stand/      800 帧安静站立 npz
/tmp/lens110_stand_old/        原始 8 段站立 npz
/tmp/lens110_xml_with_floor.bak
```

/tmp 重启会清空。需要长期保留就自己拷走, 或者让我搬进工程目录。

## 复现命令

```bash
conda activate mjlab
cd projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab
python scripts/train.py Lens110-AMP-GetUp --env.scene.num-envs 2048 \
  --agent.max-iterations 60000 --agent.save-interval 500 --enable-nan-guard True
python scripts/play.py Lens110-AMP-GetUp \
  --checkpoint-file logs/rsl_rl/getup_60k/2026-09-02_18-28-52/model_54500.pt --num-envs 8
```

## 本包里的训练数据

`logs/rsl_rl/getup_60k/2026-09-02_18-28-52/` 下只放了:
`params/env.yaml`、`params/agent.yaml`、`model_0.pt`、`model_50000.pt`、`model_54500.pt`、
`export/*.onnx`、`events.out.tfevents.*`(训练曲线)。
**未放**: 中间 107 个 checkpoint(每个 22 MB, 合计约 2.3 GB)和另外几个实验 run
(`getup_60k/2026-09-03_11-52-26_rec3`、`getup_v3/`)。它们仍在原机上, 需要的话再单独打包。

## 一句话结论

这版权重会站稳、能抗推, 但**没有真正学会从倒地起身**(实测 256 个环境从 Recovery 倒地帧出生、
4 秒内站起 3.5%)。原因是这套 AMP 框架里 motion 只当"判别器正样本 + reset 出生池",
没有任何一项在跟踪 demo 的时序; 而 `track_root_height` 又要求"先触发终止才有分",
Recovery 帧大半不触发 → 起身几乎没拿到过梯度。要"按 npz 的姿势起身"得换
mjlab 的 tracking 任务族。
