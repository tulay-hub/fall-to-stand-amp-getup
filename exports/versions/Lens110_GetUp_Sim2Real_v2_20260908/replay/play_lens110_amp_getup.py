#!/usr/bin/env python
"""Lens110「倒地自动起身」策略离线验证 (MuJoCo + ONNX, 不需要实机, 不需要动作参考)。

策略观测是纯本体 288 维 (4 帧 time-major × 72), 见 docs/OBSERVATION_ORDER.md。
本脚本的观测拼装/关节顺序/action 公式已逐项与 mjlab 训练环境的 obs 对照过,
并验证: 理想站姿出生可站稳 (末段 93% 时间骨盆 > 0.62 m), 从 Recovery 倒地帧出生则停在
约 0.355 m 起不来 —— 与 docs/KNOWN_ISSUES.md 第 1 条一致。

用法 (conda activate gmr):
    python play_lens110_amp_getup.py                         # 理想站姿出生
    python play_lens110_amp_getup.py --init stand            # 从训练出生帧(站立段)出生
    python play_lens110_amp_getup.py --init recovery --lowest  # 从倒地最低帧出生
    python play_lens110_amp_getup.py --init default --push-every 2.0   # 每 2 秒推一次

窗口按键: 空格 暂停 | r 重新出生 | f 推一下 | n 切换出生方式 | [ 减速 | ] 加速 | Esc 退出
"""

import argparse
import collections
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
INIT_MODES = ("default", "stand", "recovery")


def quat_wxyz_from_axis_angle(a):
    ang = float(np.linalg.norm(a))
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.asarray(a, float) / ang
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "..", "policy",
                                                   "Lens110-AMP-GetUp_model_72500.onnx"))
    ap.add_argument("--cfg", default=os.path.join(HERE, "..", "config", "deploy_config.yaml"))
    ap.add_argument("--mjcf", default=os.path.join(HERE, "mjcf", "lens110.xml"))
    ap.add_argument("--init-store", default=os.path.join(HERE, "..", "assets", "init_states"))
    ap.add_argument("--init", default="default", choices=INIT_MODES, help="出生方式")
    ap.add_argument("--frame", type=int, default=0, help="出生帧序号 (stand/recovery)")
    ap.add_argument("--lowest", action="store_true", help="取该动作里骨盆最低的帧")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--push-every", type=float, default=0.0, help=">0 时每隔 N 秒随机推一次")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    joints = cfg["joint_names_mjcf"]
    deg = np.array(cfg["default_joint_pos_rad"], float)
    kp = np.array(cfg["joint_stiffness_kp"], float)
    kd = np.array(cfg["joint_damping_kd"], float)
    ue = np.array(cfg["effort_limit_nm"], float)
    scale = np.array(cfg["action_scale"], float)
    dt = float(cfg["frequencies"]["timestep"])
    decim = int(cfg["frequencies"]["decimation"])
    n_hist = int(cfg["observation"]["history"])
    policy_dt = dt * decim

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joints])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joints])
    free = next(i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    pelvis = model.jnt_bodyid[free]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if gid < 0:
        raise RuntimeError("MJCF 里没有 imu_ang_vel 陀螺传感器, 无法复现 base_ang_vel 观测")
    gadr, gdim = int(model.sensor_adr[gid]), int(model.sensor_dim[gid])

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    prev = {"a": np.zeros(21)}

    def obs_block():
        """当前帧的 72 维观测, 顺序与 mjlab actor 组一致。"""
        ang = np.asarray(data.sensordata[gadr:gadr + gdim], float)[:3]      # 骨盆系角速度
        grav = data.xmat[pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([ang, grav, np.zeros(3),
                               data.qpos[qadr] - deg, data.qvel[vadr], prev["a"]]).astype(np.float32)

    def init_state(mode, frame, lowest):
        if mode == "default":
            return np.array([0.0, 0.0, float(cfg["standing_pelvis_height_m"])]), \
                np.array([1.0, 0, 0, 0]), deg.copy(), np.zeros(21)
        path = os.path.join(args.init_store, "stand_from_motion.npz" if mode == "stand"
                            else "fall_recovery_00.npz")
        if not os.path.exists(path):
            raise SystemExit(f"缺少出生文件: {path}")
        z = np.load(path)
        jn = list(z["joint_names"])
        q = np.array([z["joint_pos"][0 if lowest else frame][jn.index(n)] for n in joints], float)
        k = int(np.argmin(z["body_pos_w"][:, 0, 2])) if lowest else min(frame, len(q) - 1)
        q = np.array([z["joint_pos"][k][jn.index(n)] for n in joints], float)
        pos = np.asarray(z["body_pos_w"][k, 0], float)
        quat = np.asarray(z["body_quat_w"][k, 0], float)
        jvel = np.array([z["joint_vel"][k][jn.index(n)] for n in joints], float)
        return pos, quat, q, jvel

    def respawn():
        pos, quat, q0, jv0 = init_state(st["init"], args.frame, args.lowest)
        data.qpos[:] = 0.0
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
        data.qpos[qadr] = q0
        data.qvel[:] = 0.0
        data.qvel[vadr] = jv0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        prev["a"] = np.zeros(21)
        one = obs_block()
        st["hist"] = collections.deque([one.copy() for _ in range(n_hist)], maxlen=n_hist)
        st["t0"] = time.time()
        st["up_frames"] = 0
        st["total_frames"] = 0

    st = {"paused": False, "respawn": True, "init": args.init, "hist": None,
          "t0": 0.0, "speed": args.speed, "up_frames": 0, "total_frames": 0, "next_push": 0.0}

    def push_once():
        data.qvel[0:3] += np.random.uniform([-0.6, -0.4, -0.2], [0.6, 0.4, 0.2])
        data.qvel[3:6] += np.random.uniform(-0.4, 0.4, 3)

    def on_key(k):
        if k == 32:
            st["paused"] = not st["paused"]
        elif k in (ord("r"), ord("R")):
            st["respawn"] = True
        elif k in (ord("f"), ord("F")):
            push_once()
        elif k in (ord("n"), ord("N")):
            st["init"] = INIT_MODES[(INIT_MODES.index(st["init"]) + 1) % len(INIT_MODES)]
            st["respawn"] = True
            print(f"[replay] 出生方式 -> {st['init']}")
        elif k == ord("["):
            st["speed"] = max(0.05, st["speed"] * 0.8)
        elif k == ord("]"):
            st["speed"] = min(4.0, st["speed"] * 1.25)

    respawn()
    print(f"[replay] onnx={os.path.basename(args.onnx)} init={args.init} dt={dt}s "
          f"decimation={decim} -> 策略 {1/policy_dt:.0f}Hz, 物理 {1/dt:.0f}Hz")
    print("[replay] 空格暂停 r重出生 f推一下 n切出生方式 [ ] 调速")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if st["respawn"]:
                respawn()
                st["respawn"] = False
            if st["paused"]:
                time.sleep(0.02)
                continue

            obs = np.concatenate(list(st["hist"]))
            action = sess.run(None, {iname: obs[None, :]})[0][0].astype(np.float64)
            prev["a"] = action
            q_des = deg + scale * action
            for _ in range(decim):
                tau = np.clip(kp * (q_des - data.qpos[qadr]) - kd * data.qvel[vadr], -ue, ue)
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[vadr] = tau
                mujoco.mj_step(model, data)
            st["hist"].append(obs_block())

            now = time.time() - st["t0"]
            st["total_frames"] += 1
            st["up_frames"] += int(data.qpos[2] > 0.62)
            if args.push_every > 0 and now >= st["next_push"]:
                push_once()
                st["next_push"] = now + args.push_every
            if st["total_frames"] % 50 == 0:
                print(f"[replay] t={now:5.1f}s 骨盆 {data.qpos[2]:.3f} m | 站起占比 "
                      f"{100*st['up_frames']/st['total_frames']:5.1f}% | 关节速度峰值 "
                      f"{np.abs(data.qvel[vadr]).max():.1f} rad/s | 出生={st['init']} "
                      f"| {st['speed']:.2f}x")
            viewer.sync()
            time.sleep(max(0.0, policy_dt / max(st["speed"], 1e-3) - (time.time() - t0)))


if __name__ == "__main__":
    main()
