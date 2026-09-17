# 摔倒恢复动作（100Hz）

来源: `walk0821/walk0821/lens110_fall*.pkl`（原始 50Hz，2575 帧）

> 重要: 这里所有 fall / fall2 / fall3 / fall4 系列都来自**同一个**源动作
> `296fdbf1-24a7-4413-b67b-48acd7953275.npz`（51.48s @50Hz），只是同一条修复链上的
> 不同迭代版本，不是 4 个不同动作。训练时同一动作只喂 1 份。

## 转换说明

- 统一重采样到 100Hz，每段 5149 帧（50Hz 两倍插值）
- root_pos / dof_pos 线性插值，root_rot 用 slerp（wxyz，已处理静态帧除零）
- 元数据（contact_validation / dof_names / retarget_info 等）原样保留
- 文件名在原名后加 `_100hz`

### 已知遗留问题

重采样时只更新了 `root_pos` / `root_rot` / `dof_pos`，`motor_dof_pos` 和
`key_body_pos` 仍是 2575 帧（50Hz 长度），**不要用这两个字段**；训练 npz 转换由
`dof_pos` + FK 重算。`lens110_fall4_v4_dejerk_100hz.pkl` 已修正该问题。

## 当前文件（已清理旧版，2026-09-01）

保留 6 个 + 1 个去抖动版：

- `lens110_fall_100hz.pkl` / `lens110_fall2_100hz.pkl` / `lens110_fall3_100hz.pkl` / `lens110_fall4_100hz.pkl`
  - 同一段动作的重采样版（关节角几乎完全相同，只差整体贴地高度，互为冗余）
- `lens110_fall4_self_collision_pairs_center_v4_100hz.pkl`
  - 综合修复链最终版（自碰撞分离 + 地面 + 硬地板，修复链末端），**训练用母本**
- `lens110_fall4_v4_dejerk_100hz.pkl`
  - 在 center_v4 上做去抖动（修抽搐）后的版本，2026-09-02 生成，推荐训练用
  - 6Hz 零相位 Butterworth 底平滑 + 超限窗口局部递降重平滑(5→1.5Hz, 边缘交叉淡化)
    + 对称速度投影(关节 10.4 rad/s, root 1.6 m/s)
  - 关节峰值速度 73.15 → 10.40 rad/s，超限帧 234 → 0；角加速度峰值 12284 → 1810 rad/s²
  - 动作保真: 21 关节平均相关系数 0.9984，关节角平均改动 0.30°，root 平均改动 0.2cm
  - 仍存在: 脚底穿地(左脚 mesh 约 -7cm)、贴地滑动(中位 0.6 m/s)、少量手臂/髋部贴近
- `lens110_fall4_ground_contact_slip_repaired_v2_100hz.pkl`
  - 防滑修复分支最终版（另一条链，未含自碰分离修正）

已删除：其余 17 个中间迭代版本（mesh_collision / static_support / exact_mesh_speed /
extreme_windows / ground_hard_floor / self_collision 各轮的旧版）。
如需找回，原始 50Hz 版本仍在 `walk0821/walk0821/lens110_fall*.pkl`。

## 待修（按优先级）

1. 脚/手穿地：左脚 mesh 最深 -72mm(2210 帧)、右肘 -89mm，需要 ankle/elbow 级别修正，
   整体抬 root 会让另一只脚悬空，不能简单做。
2. 贴地滑动：中位 0.6 m/s，需要接触锚定（可参考 slip_repaired_v2 那条分支再合并）。
3. 手臂贴近：右肘×左肘、左髋×左肘、右肘×左肩、右髋×右肘、左脚×右膝 有 <3mm 接触帧。
4. 动作是 8 段"摔→爬起"串烧、root 直线漂移近 6m，考虑裁成 1-2 个干净循环再训练。
